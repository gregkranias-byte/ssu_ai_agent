from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from openai import OpenAI
from rapidfuzz import fuzz

from prompts.templates import (
    TEMPLATE_PRE_INTRA_POST, detect_crisis, format_instructions
)

load_dotenv()

_oai = OpenAI()

def get_qdrant() -> QdrantClient:
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
        timeout=60,
    )

def embed(text: str, model: str) -> List[float]:
    return _oai.embeddings.create(model=model, input=text).data[0].embedding

def keyword_boost_score(query: str, candidate_text: str) -> float:
    # lightweight boost: fuzzy partial ratio on chunk title+text prefix
    if not query or not candidate_text:
        return 0.0
    q = query.lower().strip()
    c = candidate_text.lower()[:8000]
    return fuzz.partial_ratio(q, c) / 100.0

def retrieve(question: str, k: int = 8) -> List[Dict[str, Any]]:
    collection = os.getenv("QDRANT_COLLECTION", "ssu_notes_v1")
    emb_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    client = get_qdrant()
    qvec = embed(question, emb_model)

    # vector search
    res = client.search(
        collection_name=collection,
        query_vector=qvec,
        limit=max(30, k * 4),
        with_payload=True,
        score_threshold=None,
    )

    # keyword boost rerank
    scored = []
    for r in res:
        payload = r.payload or {}
        text = (payload.get("title","") + "\n" + payload.get("text",""))[:12000]
        boost = keyword_boost_score(question, text)
        scored.append((float(r.score) + 0.15 * boost, payload))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [p for _, p in scored[:k]]
    return top

def build_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cites = []
    seen = set()
    for c in chunks:
        for ci in (c.get("citations") or []):
            key = (ci.get("page"), tuple(c.get("section_path") or []))
            if key in seen:
                continue
            seen.add(key)
            cites.append({
                "page": ci.get("page"),
                "section_path": c.get("section_path"),
                "title": c.get("title"),
                "content_type": c.get("content_type"),
            })
    return cites

def answer(question: str, template: str = "default") -> Dict[str, Any]:
    chosen_template = TEMPLATE_PRE_INTRA_POST if template in ("default", "", None) else template
    crisis = detect_crisis(question)

    chunks = retrieve(question, k=10)
    context_blocks = []
    for c in chunks:
        sp = " > ".join(c.get("section_path") or [])
        pg = c.get("page_start")
        header = f"[{c.get('content_type')}] p{pg} — {sp}"
        body = c.get("text","")
        context_blocks.append(header + "\n" + body)

    sys = (
        "You are an anaesthesia study-note retrieval assistant. "
        "Only use the provided context. If the answer is not in the context, say so. "
        "Always include page citations like (p13) or (p13–14)."
    )
    fmt = format_instructions(chosen_template)

    user = (
        f"Question: {question}\n\n"
        f"Formatting instructions:\n{fmt}\n\n"
        "Context:\n" + "\n\n---\n\n".join(context_blocks) + "\n\n"
        "Answer now. Keep it concise and clinically oriented."
    )

    resp = _oai.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0.2,
    )

    text = resp.choices[0].message.content or ""
    out = {
        "question": question,
        "template_used": chosen_template,
        "crisis_suggested": bool(crisis and chosen_template == TEMPLATE_PRE_INTRA_POST),
        "answer": text,
        "citations": build_citations(chunks),
        "top_chunks": [
            {"title": c.get("title"), "page_start": c.get("page_start"), "section_path": c.get("section_path"), "content_type": c.get("content_type")}
            for c in chunks[:6]
        ],
    }
    return out
