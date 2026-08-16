from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.getenv("APP_DATA_DIR", str(Path(__file__).with_name("data")))).expanduser().resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "flipbook_prices.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                source_filename TEXT,
                source_hash TEXT,
                vendor TEXT,
                created_at TEXT NOT NULL,
                product_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
                vendor TEXT,
                product_name TEXT NOT NULL,
                brand TEXT,
                size TEXT,
                price_eur REAL,
                old_price_eur REAL,
                promo_condition TEXT,
                validity TEXT,
                page INTEGER,
                confidence REAL,
                normalized_name TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_products_dataset ON products(dataset_id);
            CREATE INDEX IF NOT EXISTS idx_products_price ON products(price_eur);
            CREATE INDEX IF NOT EXISTS idx_products_vendor ON products(vendor);
            CREATE INDEX IF NOT EXISTS idx_products_norm ON products(normalized_name);
            CREATE INDEX IF NOT EXISTS idx_products_confidence ON products(confidence);
            CREATE INDEX IF NOT EXISTS idx_datasets_hash ON datasets(source_hash);
            """
        )
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS product_fts USING fts5(
                    product_id UNINDEXED,
                    product_name,
                    brand,
                    vendor,
                    size,
                    tokenize='unicode61 remove_diacritics 2'
                )
                """
            )
        except sqlite3.OperationalError:
            # Some minimal SQLite builds omit FTS5. Search still works via indexed LIKE fallback.
            pass


def normalize_text(value: str | None) -> str:
    value = (value or "").lower().replace("ß", "ss")
    value = re.sub(r"[^a-z0-9äöü]+", " ", value)
    return " ".join(value.split())


def file_hash(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def dataset_name_exists(name: str) -> bool:
    with get_connection() as conn:
        row = conn.execute("SELECT 1 FROM datasets WHERE name = ? COLLATE NOCASE", (name.strip(),)).fetchone()
        return row is not None


def find_dataset_by_hash(source_hash: str):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, name, source_filename, created_at, product_count FROM datasets WHERE source_hash=? ORDER BY id DESC LIMIT 1",
            (source_hash,),
        ).fetchone()
        return dict(row) if row else None


def add_dataset(name: str, source_filename: str, source_hash: str, products: pd.DataFrame, vendor_override: str = "") -> int:
    name = name.strip()
    if not name:
        raise ValueError("Dataset name cannot be empty.")
    if products.empty:
        raise ValueError("There are no products to store.")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    vendor = vendor_override.strip() or str(products["vendor"].dropna().iloc[0] if "vendor" in products and not products["vendor"].dropna().empty else "")

    rows = []
    for _, r in products.iterrows():
        p_name = str(r.get("product_name", "") or "").strip()
        if not p_name:
            continue
        row_vendor = vendor_override.strip() or str(r.get("vendor", "") or "")
        rows.append((
            row_vendor,
            p_name,
            str(r.get("brand", "") or ""),
            str(r.get("size", "") or ""),
            _float_or_none(r.get("price_eur")),
            _float_or_none(r.get("old_price_eur")),
            str(r.get("promo_condition", "") or ""),
            str(r.get("validity", "") or ""),
            _int_or_none(r.get("page")),
            _float_or_none(r.get("confidence")),
            normalize_text(p_name),
        ))

    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO datasets(name, source_filename, source_hash, vendor, created_at, product_count) VALUES(?,?,?,?,?,?)",
            (name, source_filename, source_hash, vendor, now, len(rows)),
        )
        dataset_id = int(cur.lastrowid)
        cur.executemany(
            """
            INSERT INTO products(
                dataset_id, vendor, product_name, brand, size, price_eur, old_price_eur,
                promo_condition, validity, page, confidence, normalized_name
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [(dataset_id, *r) for r in rows],
        )
        try:
            conn.execute(
                """
                INSERT INTO product_fts(product_id, product_name, brand, vendor, size)
                SELECT id, product_name, brand, vendor, size FROM products WHERE dataset_id=?
                """,
                (dataset_id,),
            )
        except sqlite3.OperationalError:
            pass
    return dataset_id


def list_datasets() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT id, name, vendor, source_filename, created_at, product_count FROM datasets ORDER BY id DESC",
            conn,
        )


def rename_dataset(dataset_id: int, new_name: str) -> None:
    new_name = new_name.strip()
    if not new_name:
        raise ValueError("Dataset name cannot be empty.")
    with get_connection() as conn:
        conn.execute("UPDATE datasets SET name=? WHERE id=?", (new_name, dataset_id))


def delete_dataset(dataset_id: int) -> None:
    with get_connection() as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM products WHERE dataset_id=?", (dataset_id,)).fetchall()]
        if ids:
            try:
                conn.executemany("DELETE FROM product_fts WHERE product_id=?", [(i,) for i in ids])
            except sqlite3.OperationalError:
                pass
        conn.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))


def get_products(dataset_id: int | None = None, min_confidence: float = 0.0) -> pd.DataFrame:
    sql = """
        SELECT p.id, d.name AS dataset, p.vendor, p.product_name, p.brand, p.size,
               p.price_eur, p.old_price_eur, p.promo_condition, p.validity,
               p.page, p.confidence
        FROM products p JOIN datasets d ON d.id=p.dataset_id
        WHERE COALESCE(p.confidence,0) >= ?
    """
    params: list = [min_confidence]
    if dataset_id is not None:
        sql += " AND p.dataset_id=?"
        params.append(dataset_id)
    sql += " ORDER BY d.id DESC, p.page, p.product_name"
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def search_products(query: str, dataset_id: int | None = None, min_confidence: float = 0.0, limit: int = 500) -> pd.DataFrame:
    query = query.strip()
    if not query:
        return pd.DataFrame()

    tokens = [t for t in normalize_text(query).split() if len(t) >= 2]
    if not tokens:
        return pd.DataFrame()

    fts_query = " AND ".join(f'"{t}"*' for t in tokens)
    params: list = [fts_query, min_confidence]
    dataset_clause = ""
    if dataset_id is not None:
        dataset_clause = " AND p.dataset_id=?"
        params.append(dataset_id)
    params.append(limit)

    sql = f"""
        SELECT p.id, d.name AS dataset, p.vendor, p.product_name, p.brand, p.size,
               p.price_eur, p.old_price_eur, p.promo_condition, p.validity,
               p.page, p.confidence
        FROM product_fts f
        JOIN products p ON p.id = CAST(f.product_id AS INTEGER)
        JOIN datasets d ON d.id = p.dataset_id
        WHERE product_fts MATCH ? AND COALESCE(p.confidence,0) >= ? {dataset_clause}
        ORDER BY CASE WHEN p.price_eur IS NULL THEN 1 ELSE 0 END, p.price_eur ASC, bm25(product_fts)
        LIMIT ?
    """
    with get_connection() as conn:
        try:
            return pd.read_sql_query(sql, conn, params=params)
        except (sqlite3.OperationalError, pd.errors.DatabaseError):
            # Fast-enough fallback for SQLite builds without FTS5.
            like_terms = [f"%{t}%" for t in tokens]
            where = " AND ".join("p.normalized_name LIKE ?" for _ in like_terms)
            fallback_params: list = like_terms + [min_confidence]
            ds_clause = ""
            if dataset_id is not None:
                ds_clause = " AND p.dataset_id=?"
                fallback_params.append(dataset_id)
            fallback_params.append(limit)
            fallback = f"""
                SELECT p.id, d.name AS dataset, p.vendor, p.product_name, p.brand, p.size,
                       p.price_eur, p.old_price_eur, p.promo_condition, p.validity,
                       p.page, p.confidence
                FROM products p JOIN datasets d ON d.id=p.dataset_id
                WHERE {where} AND COALESCE(p.confidence,0)>=? {ds_clause}
                ORDER BY CASE WHEN p.price_eur IS NULL THEN 1 ELSE 0 END, p.price_eur ASC
                LIMIT ?
            """
            return pd.read_sql_query(fallback, conn, params=fallback_params)


def stats() -> dict:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT (SELECT COUNT(*) FROM datasets), (SELECT COUNT(*) FROM products), (SELECT COUNT(DISTINCT vendor) FROM products WHERE vendor <> '')"
        ).fetchone()
        return {"datasets": row[0], "products": row[1], "vendors": row[2]}


def _float_or_none(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value):
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
