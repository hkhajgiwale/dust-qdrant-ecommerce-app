# collection_manager.py
"""
Collection creation + optimization helpers for the 'products' collection.

Features:
- ensure_products_collection(): create named-vectors collection (text + image)
- add_payload_indexes(): create payload indexes for common fields
- set_hnsw_params(): update HNSW config (best-effort)
- enable_quantization_inplace(): try to enable quantization without recreating
- recreate_collection_with_quantization(): destructive recreate with quantization (if desired)

Usage examples (from shell or main.py):
  python -c "from collection_manager import ensure_products_collection; ensure_products_collection()"
  python -c "from collection_manager import add_payload_indexes; add_payload_indexes()"
  python -c "from collection_manager import set_hnsw_params; set_hnsw_params(m=32, ef_construct=512)"
  python -c "from collection_manager import enable_quantization_inplace; enable_quantization_inplace()"
  python -c "from collection_manager import recreate_collection_with_quantization; recreate_collection_with_quantization(force_delete=True)"
"""

from typing import Dict, Any, Optional
import traceback

from client import qdrant_client
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff
from embedder import get_text_embedding_dimension, get_image_embedding_dimension

# Quantization imports are optional / may vary across qdrant-client versions
try:
    from qdrant_client.models import QuantizationConfig, ScalarQuantization
except Exception:
    QuantizationConfig = None
    ScalarQuantization = None

# Payload schema
try:
    from qdrant_client.models import PayloadSchemaType
except Exception:
    PayloadSchemaType = None

COLLECTION_NAME = "products"


def ensure_products_collection(collection_name: str = COLLECTION_NAME, recreate_if_mismatch: bool = False):
    """
    Ensure the 'products' collection exists with named vectors 'text' and 'image'.
    - If the collection does not exist, it will be created using detected dims.
    - If collection exists but dims don't match and recreate_if_mismatch=True, it will be deleted & recreated.
    """
    text_dim = get_text_embedding_dimension()
    image_dim = get_image_embedding_dimension()

    print(f"[collection_manager] Desired schema: text_dim={text_dim}, image_dim={image_dim}")

    vectors_config = {
        "text": VectorParams(size=text_dim, distance=Distance.COSINE),
        "image": VectorParams(size=image_dim, distance=Distance.COSINE),
    }

    try:
        exists = qdrant_client.collection_exists(collection_name)
    except Exception as e:
        print("[collection_manager] ERROR: Failed to check collection existence:", e)
        traceback.print_exc()
        raise

    if not exists:
        print(f"[collection_manager] Creating collection '{collection_name}' ...")
        try:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                hnsw_config=HnswConfigDiff(m=16, ef_construct=256),
            )
            print(f"[collection_manager] Created collection '{collection_name}' with vectors: {vectors_config}")
        except Exception as e:
            print("[collection_manager] ERROR: Failed to create collection:", e)
            traceback.print_exc()
            raise
    else:
        # If exists, validate dims if possible (best-effort)
        print(f"[collection_manager] Collection '{collection_name}' already exists.")
        try:
            info = qdrant_client.get_collection(collection_name)
            existing_vectors = getattr(info, "vectors", None) or getattr(info, "vectors_count", None) or None
            # We don't rely on exact returned shape (varies by server/client); just print info for debugging
            print(f"[collection_manager] Remote collection info (abridged): {type(info)}")
        except Exception:
            # get_collection may not exist in some client versions
            pass

        if recreate_if_mismatch:
            # Optionally attempt to check and recreate if vector config mismatch is found.
            # Note: detecting mismatch reliably across client/server versions is tricky,
            # so we only do this when the user explicitly asks.
            print("[collection_manager] recreate_if_mismatch=True, attempting to recreate collection for exact schema.")
            try:
                qdrant_client.delete_collection(collection_name=collection_name)
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_config,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=256),
                )
                print("[collection_manager] Recreated collection with desired schema.")
            except Exception as e:
                print("[collection_manager] ERROR: failed to recreate collection:", e)
                traceback.print_exc()


def add_payload_indexes(collection_name: str = COLLECTION_NAME):
    """
    Add payload indexes for common fields to enable fast filtering.
    Non-destructive.
    """
    print(f"[collection_manager] Adding payload indexes to '{collection_name}' ...")
    if PayloadSchemaType is None:
        print("  - PayloadSchemaType not available in this qdrant-client. Skipping payload index creation.")
        return

    indexes = {
        "price": PayloadSchemaType.FLOAT,
        "category": PayloadSchemaType.KEYWORD,
        "brand": PayloadSchemaType.KEYWORD,
        "availability": PayloadSchemaType.KEYWORD,
        "rating": PayloadSchemaType.FLOAT,
    }

    for field, schema in indexes.items():
        try:
            qdrant_client.create_payload_index(collection_name=collection_name, field_name=field, field_schema=schema)
            print(f"  - Created payload index for '{field}' ({schema}).")
        except Exception as e:
            # If already exists or unsupported, show a friendly message
            print(f"  - Could not create payload index for '{field}': {e}")


def set_hnsw_params(collection_name: str = COLLECTION_NAME, m: int = 16, ef_construct: int = 256):
    """
    Update HNSW params on the collection (best-effort).
    Some changes may require recreating the collection or server-side migration.
    """
    print(f"[collection_manager] Setting HNSW params: m={m}, ef_construct={ef_construct} for '{collection_name}'")
    try:
        diff = HnswConfigDiff(m=m, ef_construct=ef_construct)
        qdrant_client.update_collection(collection_name=collection_name, hnsw_config=diff)
        print("[collection_manager] update_collection request submitted for HNSW params.")
    except Exception as e:
        print("[collection_manager] Failed to update HNSW params:", e)
        traceback.print_exc()


def enable_quantization_inplace(collection_name: str = COLLECTION_NAME, scalar_type: str = "INT8"):
    """
    Try to enable quantization in-place (best-effort).
    If unsupported by client/server, this will emit an informative error.
    """
    if QuantizationConfig is None or ScalarQuantization is None:
        print("[collection_manager] Quantization classes not available in this qdrant-client version.")
        print("  -> To enable quantization, upgrade qdrant-client and server, or use recreate_collection_with_quantization().")
        return

    print(f"[collection_manager] Attempting to enable quantization in-place (type={scalar_type}) on '{collection_name}' ...")
    try:
        quant_conf = QuantizationConfig(scalar=ScalarQuantization(type=scalar_type))
        qdrant_client.update_collection(collection_name=collection_name, quantization_config=quant_conf)
        print("[collection_manager] quantization update request sent. Check server logs or collection info to confirm.")
    except Exception as e:
        print("[collection_manager] In-place quantization failed:", e)
        traceback.print_exc()
        print("  -> Use recreate_collection_with_quantization(force_delete=True) to recreate collection with quantization.")


def recreate_collection_with_quantization(collection_name: str = COLLECTION_NAME, force_delete: bool = False, scalar_type: str = "INT8"):
    """
    Recreate the collection with quantization. WARNING: destructive (deletes data).
    Use only if you can re-ingest.
    """
    print(f"[collection_manager] recreate_collection_with_quantization: collection={collection_name}, force_delete={force_delete}, scalar_type={scalar_type}")
    text_dim = get_text_embedding_dimension()
    image_dim = get_image_embedding_dimension()

    vectors_config = {
        "text": VectorParams(size=text_dim, distance=Distance.COSINE),
        "image": VectorParams(size=image_dim, distance=Distance.COSINE),
    }

    if qdrant_client.collection_exists(collection_name):
        if not force_delete:
            raise RuntimeError(f"Collection '{collection_name}' exists. Call with force_delete=True to drop it.")
        print(f"[collection_manager] Deleting existing collection '{collection_name}' ...")
        qdrant_client.delete_collection(collection_name=collection_name)

    # If quantization model classes exist, try to use them; else create without quantization as fallback
    if QuantizationConfig is not None and ScalarQuantization is not None:
        try:
            quant_conf = QuantizationConfig(scalar=ScalarQuantization(type=scalar_type))
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                hnsw_config=HnswConfigDiff(m=16, ef_construct=256),
                quantization_config=quant_conf,
            )
            print("[collection_manager] Created collection with quantization.")
            return
        except Exception as e:
            print("[collection_manager] Failed to create quantized collection:", e)
            traceback.print_exc()
            print("[collection_manager] Falling back to creating collection without quantization.")
    # Fallback: create without quantization
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=256),
        )
        print("[collection_manager] Created collection without quantization (fallback).")
    except Exception as e:
        print("[collection_manager] Failed to create collection in fallback path:", e)
        traceback.print_exc()
        raise


# CLI-style runner when invoked directly
if __name__ == "__main__":
    import sys

    def help_text():
        print("collection_manager CLI")
        print("  ensure_products_collection([recreate_if_mismatch])")
        print("  add_payload_indexes()")
        print("  set_hnsw_params(m, ef_construct)")
        print("  enable_quantization_inplace()")
        print("  recreate_collection_with_quantization(force_delete=True)")
        print("")
        print("Examples:")
        print("  python -c \"from collection_manager import ensure_products_collection; ensure_products_collection()\"")
        print("  python -c \"from collection_manager import add_payload_indexes; add_payload_indexes()\"")
        print("  python -c \"from collection_manager import enable_quantization_inplace; enable_quantization_inplace()\"")

    if len(sys.argv) <= 1:
        help_text()
        sys.exit(0)

    cmd = sys.argv[1]

    try:
        if cmd == "ensure":
            recreate_flag = False
            if len(sys.argv) > 2 and sys.argv[2].lower() in ("true", "1", "yes"):
                recreate_flag = True
            ensure_products_collection(recreate_if_mismatch=recreate_flag)
        elif cmd == "add_payload_indexes":
            add_payload_indexes()
        elif cmd == "set_hnsw":
            m = int(sys.argv[2]) if len(sys.argv) > 2 else 16
            ef = int(sys.argv[3]) if len(sys.argv) > 3 else 256
            set_hnsw_params(m=m, ef_construct=ef)
        # elif cmd == "quantize_inplace":
        #     enable_quantization_inplace()
        # elif cmd == "recreate_quantized":
        #     recreate_collection_with_quantization(force_delete=True)
        else:
            print("Unknown command:", cmd)
            help_text()
    except Exception as e:
        print("Error running command:", e)
        traceback.print_exc()
