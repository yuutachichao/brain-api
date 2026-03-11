import os
import uuid
import math
from typing import Any, Dict, List, Optional

import psycopg
import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="brain-api", version="0.1.0")

POSTGRES_URL = os.environ.get("POSTGRES_URL", "")
QDRANT_URL = os.environ.get("QDRANT_URL", "")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "")
API_KEY = os.environ.get("API_KEY", "")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "ollama")
EMBEDDING_VERSION = os.environ.get("EMBEDDING_VERSION", "v1")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "knowledge_bge_m3_v1")
TOP_K_DEFAULT = int(os.environ.get("TOP_K_DEFAULT", "8"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))


def check_auth(auth_header: Optional[str]):
expected = f"Bearer {API_KEY}" if API_KEY else None
if expected and auth_header != expected:
raise HTTPException(status_code=401, detail="unauthorized")


def clean_text(text: str) -> str:
return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
if len(text) <= size:
return [text]
chunks = []
start = 0
while start < len(text):
end = start + size
chunks.append(text[start:end])
if end >= len(text):
break
start = max(end - overlap, start + 1)
return chunks


def get_conn():
if not POSTGRES_URL:
raise HTTPException(status_code=500, detail="POSTGRES_URL missing")
return psycopg.connect(POSTGRES_URL)


def ensure_qdrant_collection(dim: int):
url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}"
res = requests.get(url, timeout=20)
if res.status_code == 200:
return
payload = {"vectors": {"size": dim, "distance": "Cosine"}}
res = requests.put(url, json=payload, timeout=20)
res.raise_for_status()


def embed(text: str) -> List[float]:
url = f"{OLLAMA_URL}/api/embeddings"
res = requests.post(url, json={"model": EMBEDDING_MODEL, "prompt": text}, timeout=120)
res.raise_for_status()
data = res.json()
if "embedding" not in data:
raise HTTPException(status_code=500, detail=f"embedding failed: {data}")
return data["embedding"]


class IngestRequest(BaseModel):
title: Optional[str] = None
source_url: Optional[str] = None
source_type: str = "web"
author: Optional[str] = None
language: str = "zh-TW"
raw_content: str
summary: Optional[str] = None
key_points: List[str] = Field(default_factory=list)
tags: List[str] = Field(default_factory=list)
assistant_notes: Optional[str] = None


class SearchRequest(BaseModel):
query: str
top_k: int = TOP_K_DEFAULT
tags: List[str] = Field(default_factory=list)


@app.get("/health")
def health():
return {"ok": True}


@app.post("/ingest/article")
def ingest_article(req: IngestRequest, authorization: Optional[str] = Header(default=None)):
check_auth(authorization)
clean = clean_text(req.raw_content)
chunks = chunk_text(clean, CHUNK_SIZE, CHUNK_OVERLAP)
if not chunks:
raise HTTPException(status_code=400, detail="empty content")

first_vec = embed(chunks[0])
dim = len(first_vec)
ensure_qdrant_collection(dim)

article_id = str(uuid.uuid4())

with get_conn() as conn:
with conn.cursor() as cur:
cur.execute(
"""
insert into articles (id, title, source_url, source_type, author, language, raw_content, clean_content, summary, key_points, tags, assistant_notes, status)
values (%s, %s, %s, %s, %s, %s, %s,
hunk_id": chunk_id,
"chunk_index": idx,
"title": req.title,
"source_url": req.source_url,
"tags": req.tags,
"language": req.language,
"embedding_model": EMBEDDING_MODEL,
"embedding_version": EMBEDDING_VERSION,
}
qres = requests.put(
f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points",
json={"points": [{"id": point_id, "vector": vec, "payload": payload}]},
timeout=60,
)
qres.raise_for_status()
cur.execute(
"""
insert into article_chunks (id, article_id, chunk_index, chunk_text, token_count, embedding_provider, embedding_model, embedding_dim, embedding_version, qdrant_collection, qdrant_point_id)
values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""",
(
chunk_id, article_id, idx, chunk, approx_tokens(chunk), EMBEDDING_PROVIDER,
EMBEDDING_MODEL, dim, EMBEDDING_VERSION, QDRANT_COLLECTION, point_id
),
)

cur.execute(
"insert into ingestion_logs (source_url, article_id, status, message) values (%s, %s, %s, %s)",
(req.source_url, article_id, "success", f"ingested {len(chunks)} chunks"),
)
conn.commit()

return {"ok": True, "article_id": article_id, "chunks": len(chunks), "collection": QDRANT_COLLECTION}


@app.post("/search")
def search(req: SearchRequest, authorization: Optional[str] = Header(default=None)):
check_auth(authorization)
vec = embed(req.query)
sres = requests.post(
f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
json={"vector": vec, "limit": req.top_k, "with_payload": True},
timeout=60,
)
sres.raise_for_status()
hits = sres.json().get("result", [])
article_ids = [h.get("payload", {}).get("article_id") for h in hits if h.get("payload", {}).get("article_id")]
article_map: Dict[str, Dict[str, Any]] = {}
if article_ids:
with get_conn() as conn:
with conn.cursor() as cur:
cur.execute(
"select id::text, title, source_url, summary, tags from articles where id = any(%s)",
(article_ids,),
)
for row in cur.fetchall():
article_map[row[0]] = {
"title": row[1],
"source_url": row[2],
"summary": row[3],
"tags": row[4],
}
results = []
for h in hits:
payload = h.get("payload", {})
meta = article_map.get(payload.get("article_id"), {})
results.append({
"score": h.get("score"),
"article_id": payload.get("article_id"),
"chunk_id": payload.get("chunk_id"),
"title": meta.get("title") or payload.get("title"),
"source_url": meta.get("source_url") or payload.get("source_url"),
"chunk_text": fetch_chunk_text(payload.get("chunk_id")),
"summary": meta.get("summary"),
"tags": meta.get("tags") or payload.get("tags") or [],
})
return {"ok": True, "results": results}


@app.get("/article/{article_id}")
def get_article(article_id: str, authorization: Optional[str] = Header(default=None)):
check_auth(authorization)
with get_conn() as conn:
with conn.cursor() as cur:
cur.execute(
"select id::text, title, source_url, summary, key_points, tags, assistant_notes from articles where id = %s",
(article_id,),
)
row = cur.fetchone()
if not row:
raise HTTPException(status_code=404, detail="not found")
return {"ok": True, "article": {"id": row[0], "title": row[1], "source_url": row[2], "summary": row[3], "key_points": row[4], "tags": row[5], "assistant_notes": row[6]}}


@app.get("/articles/recent")
def recent(limit: int = 20, authorization: Optional[str] = Header(default=None)):
check_auth(authorization)
with get_conn() as conn:
with conn.cursor() as cur:
cur.execute(
"select id::text, title, source_url, created_at from articles order by created_at desc limit %s",
(limit,),
)
rows = cur.fetchall()
return {"ok": True, "items": [{"id": r[0], "title": r[1], "source_url": r[2], "created_at": r[3].isoformat()} for r in rows]}


def fetch_chunk_text(chunk_id: Optional[str]) -> Optional[str]:
if not chunk_id:
return None
with get_conn() as conn:
with conn.cursor() as cur:
cur.execute("select chunk_text from article_chunks where id = %s", (chunk_id,))
row = cur.fetchone()
return row[0] if row else None


def approx_tokens(text: str) -> int:
return max(1, math.ceil(len(text) / 4))


def json_dump(items: List[str]) -> str:
import json
return json.dumps(items, en
sure_ascii=False)

