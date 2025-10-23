# api.py (normalize + dedupe final)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from client import qdrant_client
from embedder import embed_text
from qdrant_client.models import SearchParams
from qdrant_client.http.models import QueryResponse

app = FastAPI(title="Qdrant Semantic Search API (final-v2)")

class SearchQuery(BaseModel):
    query: str
    limit: int = 5
    vector_name: str = "text"
    query_filter: Optional[Dict[str, Any]] = None

class ProductItem(BaseModel):
    id: str
    title: Optional[str]
    description: Optional[str]
    price: Optional[float]
    product_url: Optional[str]
    images: Optional[List[str]]
    score: float

def _extract_points(resp) -> List[Any]:
    """Return the raw list of scored points from different resp shapes."""
    if resp is None:
        return []
    if isinstance(resp, QueryResponse):
        return resp.points or []
    if isinstance(resp, dict):
        # qdrant HTTP shape sometimes returns {"result": [...]} or {"points": [...]}
        for key in ("result", "points"):
            if key in resp:
                return resp[key] or []
        # fallback: maybe top-level list disguised as dict
        return []
    if isinstance(resp, (list, tuple)):
        return list(resp)
    # unknown shape: try to get attribute
    return getattr(resp, "points", []) or []

def _point_to_dict(p: Any) -> Dict[str, Any]:
    """Normalize a single scored point (object or dict) -> dict."""
    if isinstance(p, dict):
        pid = p.get("id") or (p.get("point") or {}).get("id")
        score = p.get("score", 0.0)
        payload = p.get("payload") or (p.get("point") or {}).get("payload") or {}
    else:
        pid = getattr(p, "id", None)
        score = getattr(p, "score", 0.0)
        payload = getattr(p, "payload", None) or {}

    return {
        "id": str(pid),
        "title": payload.get("title"),
        "description": payload.get("description"),
        "price": payload.get("price"),
        "product_url": payload.get("product_url"),
        "images": payload.get("images"),
        "score": float(score or 0.0),
    }

@app.get("/")
async def health_check():
    """Simple connection check."""
    try:
        info = qdrant_client.get_collections()
        return {"status": "ok", "collections": [c.name for c in info.collections]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant connection failed: {e}")

@app.post("/search/{collection_name}", response_model=List[ProductItem])
async def semantic_search(collection_name: str, search_query: SearchQuery):
    if not search_query.query or not search_query.query.strip():
        raise HTTPException(status_code=400, detail="Empty query not allowed.")

    # embed
    try:
        qvec = embed_text(search_query.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # query qdrant
    try:
        resp = qdrant_client.query_points(
            collection_name=collection_name,
            query=qvec,
            using=search_query.vector_name,
            limit=max(50, search_query.limit),  # fetch more to dedupe then trim
            query_filter=search_query.query_filter,
            search_params=SearchParams(exact=False),
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant query failed: {e}")

    raw_points = _extract_points(resp)

    # Normalize and dedupe by product_url (keep highest score)
    dedupe_map: Dict[str, Dict[str, Any]] = {}
    fallback_list: List[Dict[str, Any]] = []  # if no product_url present, keep order

    for p in raw_points:
        d = _point_to_dict(p)
        url = d.get("product_url")
        if url:
            existing = dedupe_map.get(url)
            if not existing or d["score"] > existing["score"]:
                dedupe_map[url] = d
        else:
            fallback_list.append(d)

    # Combine: prefer deduped values first, then fallback (both sorted by score)
    combined = list(dedupe_map.values()) + fallback_list
    # sort by score desc and limit to requested number
    combined.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    top = combined[: search_query.limit]

    return top
