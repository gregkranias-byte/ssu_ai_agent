from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from retrieval.rag import answer
from prompts.templates import TEMPLATE_PRE_INTRA_POST, TEMPLATE_CRISIS, TEMPLATE_EXAM

app = FastAPI(title="SSU Notes RAG Agent", version="1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)
    template: str = Field(default="default", description="default | pre_intra_post | crisis | exam")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask(req: AskRequest):
    template = req.template
    if template == "default":
        template = TEMPLATE_PRE_INTRA_POST
    elif template in ("pre", "pre/intra/post"):
        template = TEMPLATE_PRE_INTRA_POST
    elif template == "crisis":
        template = TEMPLATE_CRISIS
    elif template == "exam":
        template = TEMPLATE_EXAM

    return answer(req.question, template=template)
