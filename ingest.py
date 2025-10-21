# ingest.py
"""
Ingestion pipeline:
 - Fetch product page URLs from a collection page
 - For each URL: scrape product data, embed text and image, create PointStruct
 - Batch-upsert points to Qdrant (supports named vectors: 'text' and 'image')

Make sure:
 - data.py exposes: fetch_collection_product_urls(collection_url, limit) and fetch_product_data(product_url)
 - embedder.py exposes: embed_text, embed_image_from_url, get_text_embedding_dimension, get_image_embedding_dimension
 - collection_manager.ensure_products_collection() exists (creates collection with named vectors)
 - client.py exposes `qdrant_client`
"""

import uuid
import time
import traceback
from typing import List, Dict, Optional

from client import qdrant_client
from collection_manager import ensure_products_collection
from data import fetch_collection_product_urls, fetch_product_data
from embedder import embed_text, embed_image_from_url, get_text_embedding_dimension, get_image_embedding_dimension
from qdrant_client.models import PointStruct

BATCH_SIZE = 16
SLEEP_AFTER_BATCH = 0.6


def _build_point(point_id: str, text_vec: List[float], image_vec: Optional[List[float]], payload: Dict):
    """
    Build a PointStruct for Qdrant. Use named vectors inside `vector` field:
      vector = {"text": text_vec, "image": image_vec}
    If image_vec is None, only "text" is present.
    """
    vector_payload = {"text": text_vec}
    if image_vec is not None:
        vector_payload["image"] = image_vec

    p = PointStruct(
        id=point_id,
        vector=vector_payload,
        payload=payload
    )
    return p


def ingest_from_store(collection_url: str, limit: int = 50):
    """
    Top-level ingestion function.
    """
    print("Starting ingestion from:", collection_url)

    # Ensure collection exists (collection_manager should create named-vector collection)
    ensure_products_collection()

    # 1) Fetch product URLs
    try:
        product_urls = fetch_collection_product_urls(collection_url, limit=limit)
    except Exception as e:
        print("Failed to fetch product URLs from collection:", e)
        traceback.print_exc()
        return

    print(f"Found {len(product_urls)} product URLs (limited to {limit}).")

    # dims (informational)
    try:
        text_dim = get_text_embedding_dimension()
        img_dim = get_image_embedding_dimension()
        print(f"Text embedding dim = {text_dim}, image embedding dim = {img_dim}")
    except Exception as e:
        print("Warning: failed to fetch model dims:", e)

    batch: List[PointStruct] = []
    total_upserted = 0

    for idx, url in enumerate(product_urls, start=1):
        print(f"[{idx}/{len(product_urls)}] Processing: {url}")
        try:
            pdata = fetch_product_data(url)
            if not pdata:
                print("  -> fetch_product_data returned nothing, skipping.")
                continue

            title = pdata.get("title", "") or ""
            descr = pdata.get("description", "") or ""
            text_to_embed = (title + " " + descr).strip() or title or descr
            if not text_to_embed:
                # safety: set a small fallback so text_vector isn't empty
                text_to_embed = title or descr or "product"

            # embed text
            try:
                text_vec = embed_text(text_to_embed)
            except Exception as e:
                print("  -> Text embedding failed for", url, "error:", e)
                traceback.print_exc()
                continue

            # embed image (first image) if available
            image_vec = None
            imgs = pdata.get("images") or []
            if imgs:
                first_img = imgs[0]
                try:
                    image_vec = embed_image_from_url(first_img)
                except Exception as e:
                    # don't crash on image embedding failure — proceed with text only
                    print("  -> Image embedding failed for:", first_img, "error:", e)
                    # keep image_vec = None

            payload = {
                "title": title,
                "description": descr,
                "price": pdata.get("price"),
                "product_url": pdata.get("product_url"),
                "images": pdata.get("images"),
            }

            point_id = str(uuid.uuid4())
            point = _build_point(point_id, text_vec, image_vec, payload)
            batch.append(point)

        except Exception as e:
            print("  -> Unexpected error while processing url:", url, "error:", e)
            traceback.print_exc()
            continue

        # flush batch
        if len(batch) >= BATCH_SIZE:
            try:
                qdrant_client.upsert(collection_name="products", points=batch)
                total_upserted += len(batch)
                print(f"  -> Upserted batch of {len(batch)} points (total {total_upserted}).")
            except Exception as e:
                print("  -> Upsert error for batch:", e)
                traceback.print_exc()
                # try per-point fallback to detect bad point(s)
                for p in batch:
                    try:
                        qdrant_client.upsert(collection_name="products", points=[p])
                        total_upserted += 1
                    except Exception as e2:
                        print("    -> Single point upsert failed for id:", p.id, e2)
                # continue after per-point attempts
            batch = []
            time.sleep(SLEEP_AFTER_BATCH)

    # final flush
    if batch:
        try:
            qdrant_client.upsert(collection_name="products", points=batch)
            total_upserted += len(batch)
            print(f"  -> Upserted final batch of {len(batch)} points (total {total_upserted}).")
        except Exception as e:
            print("  -> Final upsert error:", e)
            traceback.print_exc()
            # per-point retry
            for p in batch:
                try:
                    qdrant_client.upsert(collection_name="products", points=[p])
                    total_upserted += 1
                except Exception as e2:
                    print("    -> Final single point upsert failed for id:", p.id, e2)

    print("Ingestion finished. Total points upserted (approx):", total_upserted)
