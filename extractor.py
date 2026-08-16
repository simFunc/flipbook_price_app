from __future__ import annotations

import io
import os
import re
import shutil
from dataclasses import dataclass
from typing import Iterable

import fitz
import pandas as pd

PROMO_PREFIXES = (
    "AKTION", "EXTREM", "AB ", "BEI ", "1 PKG", "1 FL", "1 DOSE", "1 STÜCK",
    "1 TAFEL", "1 TUBE", "BIS ", "DO & FR", "MO", "GÜLTIG", "STATT", "-",
    "ANGEBOTE GÜLTIG", "SOLANGE DER VORRAT", "DIESER ARTIKEL", "NUR KURZE ZEIT",
    "NUR FÜR KURZE ZEIT", "GROSSPACKUNG", "100%", "100 %", "2+1", "3+3", "1+1",
    "2+2", "4+2", "12+12", "MEGA-WOCHENENDE", "MARKENHIGHLIGHT"
)

KNOWN_VENDORS = ["BILLA PLUS", "BILLA", "SPAR", "INTERSPAR", "EUROSPAR", "HOFER", "LIDL", "PENNY"]

@dataclass
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    max_size: float
    spans: list

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\u00ad", "").replace("\ufffe", "").replace("\u2009", " ")).strip()


def _extract_native_lines(page: fitz.Page) -> list[Line]:
    out = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if "lines" not in block:
            continue
        for line in block["lines"]:
            spans = [s for s in line.get("spans", []) if _clean_text(s.get("text", ""))]
            if not spans:
                continue
            text = _clean_text(" ".join(s["text"] for s in spans))
            x0, y0, x1, y1 = line["bbox"]
            out.append(Line(text, x0, y0, x1, y1, max(float(s["size"]) for s in spans), spans))
    return out


class OCRUnavailableError(RuntimeError):
    """Raised when a scanned/image PDF needs OCR but no OCR engine is available."""


def _configure_tesseract():
    """Resolve Tesseract in a deployment-friendly way.

    Priority:
    1) TESSERACT_CMD environment variable
    2) executable available on PATH (the Docker image and Streamlit packages.txt install it here)
    """
    import pytesseract

    configured = os.getenv("TESSERACT_CMD", "").strip()
    if configured:
        pytesseract.pytesseract.tesseract_cmd = configured
        if os.path.isfile(configured):
            return pytesseract

    discovered = shutil.which("tesseract")
    if discovered:
        pytesseract.pytesseract.tesseract_cmd = discovered
        return pytesseract

    raise OCRUnavailableError(
        "This PDF is image/scanned and requires OCR, but Tesseract is not available in the runtime. "
        "Deploy using the included Dockerfile (recommended) or a platform that installs packages.txt."
    )


def ocr_status() -> tuple[bool, str]:
    """Return OCR availability and a short deployment-safe status message."""
    try:
        pytesseract = _configure_tesseract()
        version = str(pytesseract.get_tesseract_version()).splitlines()[0]
        return True, f"Tesseract {version}"
    except Exception as exc:
        return False, str(exc)


def _extract_ocr_lines(page: fitz.Page) -> list[Line]:
    # Free OCR fallback. In production it is installed inside the deployment image.
    pytesseract = _configure_tesseract()
    from PIL import Image

    zoom = 2.5
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--psm 11")
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRUnavailableError(
            "OCR is required for this PDF, but Tesseract could not be started. "
            "Use the included Dockerfile for deployment so OCR is installed with the app."
        ) from exc
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, txt in enumerate(d["text"]):
        if not txt.strip() or int(float(d["conf"][i])) < 25:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        groups.setdefault(key, []).append(i)
    out = []
    for inds in groups.values():
        text = _clean_text(" ".join(d["text"][i] for i in inds))
        x0 = min(d["left"][i] for i in inds) / zoom
        y0 = min(d["top"][i] for i in inds) / zoom
        x1 = max(d["left"][i] + d["width"][i] for i in inds) / zoom
        y1 = max(d["top"][i] + d["height"][i] for i in inds) / zoom
        max_size = max(d["height"][i] for i in inds) / zoom
        out.append(Line(text, x0, y0, x1, y1, max_size, []))
    return out


def extract_lines(page: fitz.Page) -> list[Line]:
    native = _extract_native_lines(page)
    if len(" ".join(x.text for x in native)) >= 80:
        return native
    return _extract_ocr_lines(page)


def _parse_price_candidate(line: Line):
    # Flyers often render 3.99 as two adjacent spans: large "3" + smaller "99".
    pieces = [_clean_text(s.get("text", "")) for s in line.spans] if line.spans else line.text.split()
    numeric_pieces = [re.sub(r"\D", "", p) for p in pieces if re.sub(r"\D", "", p)]
    compact = "".join(numeric_pieces)

    # Require visually prominent text to avoid package quantities becoming prices.
    if line.max_size < 20:
        return None
    if not re.fullmatch(r"\d{2,4}", compact or ""):
        return None

    # Most supermarket flyer headline prices omit the decimal separator: 399 => 3.99, 094 => 0.94.
    if len(compact) == 2:
        value = int(compact) / 100
    else:
        value = int(compact[:-2] or "0") + int(compact[-2:]) / 100
    if not (0.05 <= value <= 999.99):
        return None
    return round(value, 2)



def _product_regions_from_drawings(page: fitz.Page):
    """Find likely product-card rectangles already present in vector PDFs.

    Many retail flyers contain white rectangular product tiles. Reusing those vector
    rectangles is much more reliable (and free) than trying to infer associations
    from OCR text alone.
    """
    candidates = []
    pw, ph = page.rect.width, page.rect.height
    for d in page.get_drawings():
        fill = d.get("fill")
        r = d.get("rect")
        if not fill or r is None:
            continue
        if len(fill) < 3 or min(fill[:3]) < 0.94:
            continue
        w, h = r.width, r.height
        area = w * h
        if w < 85 or h < 60 or area > pw * ph * 0.48:
            continue
        if r.x1 < 0 or r.y1 < 0 or r.x0 > pw or r.y0 > ph:
            continue
        rr = fitz.Rect(max(0,r.x0), max(0,r.y0), min(pw,r.x1), min(ph,r.y1))
        candidates.append(rr)

    # Remove near-duplicates and rectangles almost fully contained in another tile.
    candidates.sort(key=lambda r: r.width * r.height, reverse=True)
    kept = []
    for r in candidates:
        duplicate = False
        for k in kept:
            inter = r & k
            if inter.is_empty:
                continue
            overlap = (inter.width * inter.height) / max(1, min(r.width*r.height, k.width*k.height))
            if overlap > 0.92:
                duplicate = True
                break
        if not duplicate:
            kept.append(r)
    return kept


def _record_from_region(lines, region, vendor, validity, page_num):
    card_lines = [ln for ln in lines if region.x0 <= ln.cx <= region.x1 and region.y0 <= ln.cy <= region.y1]
    price_options = []
    for ln in card_lines:
        v = _parse_price_candidate(ln)
        if v is not None:
            price_options.append((ln.max_size, ln.cy, v, ln))
    if not price_options:
        return None
    # Main offer price is normally the visually largest candidate. If tied, prefer lower placement.
    _, _, value, price_line = sorted(price_options, key=lambda z: (z[0], z[1]), reverse=True)[0]
    raw_texts = [ln.text for ln in sorted(card_lines, key=lambda z:(z.y0,z.x0)) if ln is not price_line]
    clean_lines = [ln for ln in card_lines if ln is not price_line and not _is_noise(ln.text) and ln.max_size < 22]
    # Product copy is usually near the price, but keep all clean text in a real card region.
    chosen = [ln.text for ln in sorted(clean_lines, key=lambda z:(z.y0,z.x0))]
    seen=set(); chosen=[t for t in chosen if not (t in seen or seen.add(t))]
    size = _parse_size(chosen)
    desc=[]
    for t in chosen:
        if size and size.lower() in t.lower() and len(chosen)>1:
            stripped=re.sub(re.escape(size),"",t,flags=re.I).strip(" ,-/")
            if stripped: desc.append(stripped)
        else:
            desc.append(t)
    desc=[t for t in desc if len(t)<=100][:6]
    if not desc:
        return None
    product_name=" ".join(desc).strip()
    brand=desc[0] if len(desc[0].split())<=3 else ""
    conf=0.65 + (0.1 if size else 0) + (0.05 if brand else 0) + (0.1 if len(desc)>=2 else 0)
    return {
        "vendor": vendor, "product_name": product_name, "brand": brand, "size": size,
        "price_eur": value, "old_price_eur": _parse_old_price(raw_texts),
        "promo_condition": _parse_condition(raw_texts), "validity": validity,
        "page": page_num, "confidence": round(min(conf,0.97),2)
    }

def _cluster_rows(prices: list[dict], tolerance: float = 65) -> list[list[dict]]:
    rows: list[list[dict]] = []
    for p in sorted(prices, key=lambda z: z["cy"]):
        if not rows or abs(p["cy"] - sum(x["cy"] for x in rows[-1]) / len(rows[-1])) > tolerance:
            rows.append([p])
        else:
            rows[-1].append(p)
    return rows


def _cells_for_prices(prices: list[dict], page_w: float, page_h: float):
    rows = _cluster_rows(prices)
    row_centers = [sum(p["cy"] for p in row) / len(row) for row in rows]
    cells = []
    for ri, row in enumerate(rows):
        row = sorted(row, key=lambda z: z["cx"])
        y0 = 0 if ri == 0 else (row_centers[ri - 1] + row_centers[ri]) / 2
        y1 = page_h if ri == len(rows) - 1 else (row_centers[ri] + row_centers[ri + 1]) / 2
        centers = [p["cx"] for p in row]
        for i, p in enumerate(row):
            x0 = 0 if i == 0 else (centers[i - 1] + centers[i]) / 2
            x1 = page_w if i == len(row) - 1 else (centers[i] + centers[i + 1]) / 2
            cells.append((p, (x0, y0, x1, y1)))
    return cells


def _is_noise(text: str) -> bool:
    t = _clean_text(text)
    u = t.upper()
    if not t or len(t) == 1:
        return True
    if any(u.startswith(p) for p in PROMO_PREFIXES):
        return True
    if re.fullmatch(r"[\d\s.,€%+\-/]+", t):
        return True
    if "NIEDRIGSTER 30-TAGE" in u or "PREIS ENTSPRICHT" in u:
        return True
    return False


def _parse_size(texts: Iterable[str]):
    joined = " ".join(texts)
    patterns = [
        r"\b\d+(?:[,.]\d+)?\s*(?:kg|g|ml|liter|l)\b",
        r"\b\d+\s*(?:stück|stk\.?|rollen|blatt|waschgänge)\b",
        r"\bper\s+(?:kilo|stück|100\s*g)\b",
        r"\b\d+\s*x\s*\d+(?:[,.]\d+)?\s*(?:g|ml)\b",
    ]
    for pat in patterns:
        m = re.search(pat, joined, flags=re.I)
        if m:
            return _clean_text(m.group(0))
    return ""


def _parse_old_price(texts: Iterable[str]):
    joined = " ".join(texts)
    m = re.search(r"statt\s*([0-9]+[.,][0-9]{2})", joined, flags=re.I)
    return float(m.group(1).replace(",", ".")) if m else None


def _parse_condition(texts: Iterable[str]):
    for t in texts:
        u = t.upper()
        if re.search(r"\b(?:AB|BEI)\s+\d+", u) or re.search(r"\b\d+\+\d+\b", u):
            return _clean_text(t)
    return ""


def _infer_vendor(all_text: str):
    up = all_text.upper()
    for v in KNOWN_VENDORS:
        if v in up:
            return v
    return "Unknown"


def _extract_validity(all_text: str):
    # Keep the original date wording; avoids incorrect locale assumptions.
    patterns = [
        r"GÜLTIG\s+VON\s+[^\n]{0,80}?20\d{2}",
        r"GÜLTIG\s+AM\s+[^\n]{0,80}?20\d{2}",
    ]
    for p in patterns:
        m = re.search(p, all_text, flags=re.I)
        if m:
            return _clean_text(m.group(0))
    return ""


def extract_products(pdf_bytes: bytes) -> pd.DataFrame:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_lines = []
    full_text = []
    for page in doc:
        lines = extract_lines(page)
        page_lines.append(lines)
        full_text.extend(x.text for x in lines)
    all_text = "\n".join(full_text)
    vendor = _infer_vendor(all_text)
    validity = _extract_validity(all_text)

    records = []
    for page_index, page in enumerate(doc):
        lines = page_lines[page_index]
        regions = _product_regions_from_drawings(page)
        region_records = []
        if len(regions) >= 3:
            for region in regions:
                rec = _record_from_region(lines, region, vendor, validity, page_index + 1)
                if rec:
                    region_records.append(rec)
            if region_records:
                records.extend(region_records)
                continue

        prices = []
        for li, line in enumerate(lines):
            value = _parse_price_candidate(line)
            if value is not None:
                prices.append({"line_index": li, "price": value, "cx": line.cx, "cy": line.cy, "line": line})
        if not prices:
            continue

        for p, (x0, y0, x1, y1) in _cells_for_prices(prices, page.rect.width, page.rect.height):
            # Restrict to the same visual card. Padding helps when text sits a little outside midpoint boundaries.
            pad_x = 12
            card_lines = [
                ln for ln in lines
                if (x0 - pad_x) <= ln.cx <= (x1 + pad_x)
                and y0 <= ln.cy <= y1
                and ln is not p["line"]
            ]
            card_lines = sorted(card_lines, key=lambda z: (z.y0, z.x0))
            raw_texts = [ln.text for ln in card_lines]
            clean = [ln.text for ln in card_lines if not _is_noise(ln.text) and ln.max_size < 22]

            # Prefer lines close to the price vertically; distant header/footer text is less likely to be the product.
            nearby = [ln.text for ln in card_lines if abs(ln.cy - p["cy"]) < 135 and not _is_noise(ln.text) and ln.max_size < 22]
            chosen = nearby or clean
            # Remove duplicate lines while preserving order.
            seen = set(); chosen = [t for t in chosen if not (t in seen or seen.add(t))]

            size = _parse_size(chosen)
            desc_lines = []
            for t in chosen:
                if size and size.lower() in t.lower() and len(chosen) > 1:
                    # Keep the line if it also contains useful product wording.
                    stripped = re.sub(re.escape(size), "", t, flags=re.I).strip(" ,-/")
                    if stripped:
                        desc_lines.append(stripped)
                else:
                    desc_lines.append(t)
            # Product text is usually compact; cap to avoid swallowing unrelated copy.
            desc_lines = [t for t in desc_lines if len(t) <= 100][:5]
            product_name = " ".join(desc_lines).strip()
            brand = desc_lines[0] if desc_lines and len(desc_lines[0].split()) <= 3 else ""

            # Skip obvious false detections.
            if not product_name or len(product_name) < 3:
                continue

            confidence = 0.45
            if len(desc_lines) >= 2: confidence += 0.15
            if size: confidence += 0.15
            if brand: confidence += 0.05
            if p["line"].max_size >= 28: confidence += 0.1
            confidence = min(confidence, 0.95)

            records.append({
                "vendor": vendor,
                "product_name": product_name,
                "brand": brand,
                "size": size,
                "price_eur": p["price"],
                "old_price_eur": _parse_old_price(raw_texts),
                "promo_condition": _parse_condition(raw_texts),
                "validity": validity,
                "page": page_index + 1,
                "confidence": round(confidence, 2),
            })

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["vendor","product_name","brand","size","price_eur","old_price_eur","promo_condition","validity","page","confidence"])
    # Exact duplicates can happen when a decorative duplicate price is present.
    df = df.drop_duplicates(subset=["page", "product_name", "price_eur"]).reset_index(drop=True)
    return df
