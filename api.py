from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from client import qdrant_client
from embedder import embed_text
from qdrant_client.http.models import SearchParams
from qdrant_client.models import ScoredPoint  # type hint only; actual returned objects may be dict-like

app = FastAPI(title="Qdrant Semantic Search API (fixed)")

class SearchQuery(BaseModel):
    query: str
    limit: int = 5
    vector_name: str = "text"
    query_filter: Optional[Dict[str, Any]] = None

class ProductSearchResult(BaseModel):
    id: Any
    score: float
    payload: Optional[Dict[str, Any]] = None

@app.get("/")
async def health_check():
    try:
        qdrant_client.get_collections()
        return {"status": "ok", "qdrant_connection": "successful"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant connection failed: {e}")

@app.post("/search/{collection_name}", response_model=List[ProductSearchResult])
async def semantic_search(collection_name: str, search_query: SearchQuery):
    if not search_query.query or not search_query.query.strip():
        raise HTTPException(status_code=400, detail="Empty query not allowed.")

    # 1) embed
    try:
        query_vector = embed_text(search_query.query)  # returns List[float]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # 2) call Qdrant query_points (select named vector using 'using' param)
    try:
        # NOTE: qdrant-client exposes query_points(...) which maps to POST /collections/{name}/points/query
        # The 'using' argument selects a named vector when the collection uses named vectors.
        qdrant_resp = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,                         
            using=search_query.vector_name,               
            limit=search_query.limit,
            query_filter=search_query.query_filter,       
            search_params=SearchParams(exact=False),       
            with_payload=True,                             
        )
    except TypeError as te:
        # Defensive: older/newer client versions may have slightly different fn names/args
        raise HTTPException(status_code=500, detail=f"Qdrant client call failed (type error): {te}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant query failed: {e}")

    # 3) normalize results (ScoredPoint or dict-like)
    out: List[ProductSearchResult] = []
    for item in qdrant_resp:
        # item can be ScoredPoint or dict depending on qdrant-client version/config
        try:
            pid = getattr(item, "id", None) or item.get("id") if isinstance(item, dict) else None
            score = getattr(item, "score", None) or item.get("score") if isinstance(item, dict) else None
            payload = getattr(item, "payload", None) or item.get("payload") if isinstance(item, dict) else None

            out.append(ProductSearchResult(id=pid, score=score, payload=payload))
        except Exception:
            # Last-resort fallback
            out.append(ProductSearchResult(id=str(item), score=0.0, payload=None))

    return out

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(port)
    print("Inside main function")
    host = "0.0.0.0"
    print(f"Starting uvicorn with host={host} port={port} (ENV PORT: {os.environ.get('PORT')})", flush=True)
    # Use uvicorn.run to start from python — this will show logs in Render console
    uvicorn.run("api:app", host=host, port=port, log_level="info")
