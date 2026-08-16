from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from database import (
    add_dataset,
    dataset_name_exists,
    delete_dataset,
    file_hash,
    find_dataset_by_hash,
    get_products,
    init_db,
    list_datasets,
    rename_dataset,
    search_products,
    stats,
)
from extractor import OCRUnavailableError, extract_products, ocr_status

st.set_page_config(page_title="Flipbook Price Database", page_icon="🛒", layout="wide")
init_db()


def product_table(df: pd.DataFrame, key: str):
    if df.empty:
        st.info("No products found.")
        return
    st.dataframe(
        df,
        key=key,
        use_container_width=True,
        hide_index=True,
        column_config={
            "price_eur": st.column_config.NumberColumn("Price (€)", format="%.2f"),
            "old_price_eur": st.column_config.NumberColumn("Old price (€)", format="%.2f"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.0f%%"),
        },
    )


st.title("🛒 Flipbook Price Database")
st.caption("Local, low-cost MVP: extract flyer PDFs, save datasets in SQLite, and search products quickly with SQLite FTS5.")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Upload & Extract", "Datasets", "View Data", "Search Products"],
)
ocr_ok, ocr_message = ocr_status()
if ocr_ok:
    st.sidebar.success(f"OCR ready · {ocr_message}")
else:
    st.sidebar.warning("OCR unavailable in this runtime")
st.sidebar.caption("Stored data uses the APP_DATA_DIR volume. For production, mount persistent storage there.")

if page == "Dashboard":
    s = stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Stored datasets", s["datasets"])
    c2.metric("Stored products", s["products"])
    c3.metric("Vendors", s["vendors"])

    datasets = list_datasets()
    st.subheader("Recent datasets")
    if datasets.empty:
        st.info("No datasets stored yet. Go to **Upload & Extract** to add the first flyer.")
    else:
        st.dataframe(datasets.head(10), use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Workflow:** upload PDF → extract preview → give it a dataset name → store → search across every stored flyer.

        The database uses SQLite, so there is no server bill for this MVP. Product search uses SQLite **FTS5 full-text indexing** when available instead of scanning every row one by one.
        """
    )

elif page == "Upload & Extract":
    st.header("Upload & Extract")
    uploaded = st.file_uploader("Upload supermarket flipbook PDF", type=["pdf"])

    if uploaded:
        pdf_bytes = uploaded.getvalue()
        digest = file_hash(pdf_bytes)
        duplicate = find_dataset_by_hash(digest)
        default_name = Path(uploaded.name).stem.replace("_", " ").strip()
        dataset_name = st.text_input("Dataset name", value=default_name, help="Example: BILLA Week 33 - Vienna")
        vendor_override = st.text_input("Vendor override (optional)", placeholder="e.g. BILLA, SPAR")

        if duplicate:
            st.warning(f"This exact PDF appears to already be stored as **{duplicate['name']}** with {duplicate['product_count']} products.")

        if st.button("Extract & View Products", type="primary", use_container_width=True):
            with st.spinner("Extracting and normalizing products..."):
                try:
                    df = extract_products(pdf_bytes)
                    if vendor_override.strip() and not df.empty:
                        df["vendor"] = vendor_override.strip()
                    st.session_state["extracted_products"] = df
                    st.session_state["extracted_hash"] = digest
                    st.session_state["extracted_filename"] = uploaded.name
                    st.session_state["draft_dataset_name"] = dataset_name
                    st.session_state["draft_vendor_override"] = vendor_override
                except OCRUnavailableError as exc:
                    st.error(str(exc))
                    st.info("For deployment, use the included Dockerfile. It installs OCR inside the app image, so end users do not install anything locally.")
                except Exception as exc:
                    st.exception(exc)

    if "extracted_products" in st.session_state:
        df = st.session_state["extracted_products"]
        st.subheader("Extraction preview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Products found", len(df))
        c2.metric("Pages", int(df["page"].nunique()) if not df.empty else 0)
        c3.metric("Average confidence", f"{df['confidence'].mean():.0%}" if not df.empty else "-")

        min_conf = st.slider("Preview minimum confidence", 0.0, 1.0, 0.45, 0.05)
        preview = df[df["confidence"] >= min_conf].copy() if not df.empty else df
        product_table(preview, "extract_preview")

        st.download_button(
            "Download preview CSV",
            data=preview.to_csv(index=False).encode("utf-8-sig"),
            file_name="normalized_products.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.divider()
        save_name = st.text_input("Name to store this dataset as", value=st.session_state.get("draft_dataset_name", ""), key="save_dataset_name")
        save_vendor = st.text_input("Stored vendor override (optional)", value=st.session_state.get("draft_vendor_override", ""), key="save_vendor")
        if st.button("Add / Store Dataset", use_container_width=True, disabled=df.empty):
            if not save_name.strip():
                st.error("Please enter a dataset name.")
            elif dataset_name_exists(save_name):
                st.error("A dataset with this name already exists. Choose another name or remove/rename the existing dataset.")
            else:
                try:
                    dataset_id = add_dataset(
                        save_name,
                        st.session_state.get("extracted_filename", ""),
                        st.session_state.get("extracted_hash", ""),
                        df,
                        save_vendor,
                    )
                    st.success(f"Stored **{save_name}** with {len(df)} products. Dataset ID: {dataset_id}")
                    for k in ["extracted_products", "extracted_hash", "extracted_filename", "draft_dataset_name", "draft_vendor_override"]:
                        st.session_state.pop(k, None)
                except sqlite3.IntegrityError:
                    st.error("Dataset name already exists.")
                except Exception as exc:
                    st.exception(exc)

elif page == "Datasets":
    st.header("Manage Datasets")
    datasets = list_datasets()
    if datasets.empty:
        st.info("No stored datasets yet.")
    else:
        st.dataframe(datasets, use_container_width=True, hide_index=True)
        options = {f"{r['name']}  ·  {r['product_count']} products": int(r["id"]) for _, r in datasets.iterrows()}
        selected_label = st.selectbox("Select dataset", options.keys())
        selected_id = options[selected_label]
        selected_row = datasets[datasets["id"] == selected_id].iloc[0]

        c1, c2 = st.columns(2)
        with c1:
            new_name = st.text_input("Rename dataset", value=str(selected_row["name"]))
            if st.button("Rename", use_container_width=True):
                try:
                    rename_dataset(selected_id, new_name)
                    st.success("Dataset renamed.")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("That dataset name already exists.")
        with c2:
            st.write("Delete dataset")
            confirm = st.checkbox("I understand this permanently removes its stored product rows.")
            if st.button("Remove Dataset", type="secondary", use_container_width=True, disabled=not confirm):
                delete_dataset(selected_id)
                st.success("Dataset removed.")
                st.rerun()

elif page == "View Data":
    st.header("View Stored Data")
    datasets = list_datasets()
    if datasets.empty:
        st.info("No stored data yet.")
    else:
        labels = ["All datasets"] + datasets["name"].tolist()
        selected = st.selectbox("Dataset", labels)
        dataset_id = None if selected == "All datasets" else int(datasets.loc[datasets["name"] == selected, "id"].iloc[0])
        min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05, key="view_conf")
        df = get_products(dataset_id, min_conf)
        st.caption(f"Showing {len(df):,} stored product rows")
        product_table(df, "stored_data")
        st.download_button(
            "Export displayed data as CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="stored_products.csv",
            mime="text/csv",
            use_container_width=True,
        )

elif page == "Search Products":
    st.header("Fast Product Search")
    st.caption("Search runs against the stored database. FTS5 prefix indexing keeps lookup fast even when product count grows.")
    datasets = list_datasets()
    dataset_labels = ["All datasets"] + (datasets["name"].tolist() if not datasets.empty else [])
    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input("Search product / brand", placeholder="e.g. Coca Cola, Butter, Nescafe Gold")
    with c2:
        selected = st.selectbox("Search in", dataset_labels)
    dataset_id = None
    if selected != "All datasets" and not datasets.empty:
        dataset_id = int(datasets.loc[datasets["name"] == selected, "id"].iloc[0])
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.45, 0.05, key="search_conf")

    if query.strip():
        results = search_products(query, dataset_id=dataset_id, min_confidence=min_conf, limit=1000)
        if results.empty:
            st.warning("No matching stored products found.")
        else:
            priced = results[results["price_eur"].notna()].copy()
            c1, c2, c3 = st.columns(3)
            c1.metric("Matches", len(results))
            c2.metric("Datasets matched", results["dataset"].nunique())
            c3.metric("Lowest listed price", f"€{priced['price_eur'].min():.2f}" if not priced.empty else "-")

            st.subheader("Cheapest matches first")
            product_table(results, "search_results")
            if not priced.empty:
                cheapest = priced.sort_values("price_eur").iloc[0]
                st.success(
                    f"Lowest current match: **{cheapest['product_name']}** — **€{cheapest['price_eur']:.2f}** at **{cheapest['vendor'] or 'Unknown vendor'}** in dataset **{cheapest['dataset']}**."
                )

            st.download_button(
                "Export search results",
                data=results.to_csv(index=False).encode("utf-8-sig"),
                file_name="product_search_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("Type at least part of a product or brand name to search all stored flyers.")
