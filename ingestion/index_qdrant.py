import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from openai import OpenAI


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    ap.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "ssu_notes_v1"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--embedding-model", default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"))
    args = ap.parse_args()

    qdrant_url = os.environ["QDRANT_URL"]
    qdrant_key = os.environ.get("QDRANT_API_KEY")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=120)
    oai = OpenAI()

    # create collection if missing
    if args.collection not in [c.name for c in client.get_collections().collections]:
        # embedding dim depends on model; we fetch once
        test = oai.embeddings.create(model=args.embedding_model, input="test").data[0].embedding
        dim = len(test)
        client.create_collection(
            collection_name=args.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    batch = []
    for c in iter_jsonl(Path(args.chunks)):
        batch.append(c)
        if len(batch) >= args.batch:
            upsert_batch(client, oai, args.collection, batch, args.embedding_model)
            batch = []
    if batch:
        upsert_batch(client, oai, args.collection, batch, args.embedding_model)

    print("Done.")


def upsert_batch(client: QdrantClient, oai: OpenAI, collection: str, batch, model: str):
    texts = [b["text"] for b in batch]
    embs = oai.embeddings.create(model=model, input=texts).data
    vectors = [e.embedding for e in embs]

    points = []
    for b, v in zip(batch, vectors):
        payload = {k: b.get(k) for k in [
            "chunk_id", "source_id", "doc_title", "version",
            "page_start", "page_end", "section_path",
            "specialty_tags", "periop_phase_tags",
            "content_type", "modality", "title", "citations"
        ]}
        # Keep the full text for context; if you want smaller payloads, truncate here.
        payload["text"] = b.get("text", "")[:12000]
        if b.get("content_type") in ("algorithm_image", "figure_image"):
            payload["image"] = b.get("image")

        points.append(qm.PointStruct(id=b["chunk_id"], vector=v, payload=payload))

    client.upsert(collection_name=collection, points=points)


if __name__ == "__main__":
    main()
