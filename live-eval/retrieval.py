"""Flat hybrid retrieval over one question's haystack: BM25 + Voyage dense, RRF, Jev rerank of the full shortlist.
In memory (numpy); no Postgres needed at benchmark scale."""
from __future__ import annotations

import numpy as np

import config
from text import BM25, Chunk, rrf, tokenize


def retrieve(question: str, chunks: list[Chunk], embedder, reranker) -> tuple[list[Chunk], dict]:
    if not chunks:
        return [], {"shortlist": 0}
    by_id = {c.cid: c for c in chunks}
    docs = [c.render() for c in chunks]

    bm = BM25([tokenize(d) for d in docs])
    bm_ids = [chunks[i].cid for i in bm.topk(question, config.BM25_TOP)]

    mat = embedder.embed(docs, "document")
    q = embedder.embed([question], "query")[0]
    mat_n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    q_n = q / (np.linalg.norm(q) + 1e-9)
    sims = mat_n @ q_n
    dense_ids = [chunks[i].cid for i in np.argsort(-sims)[: config.DENSE_TOP]]

    fused = rrf([bm_ids, dense_ids])[: config.SHORTLIST]
    cands = [{"id": cid, "passage": by_id[cid].render()[:2000]} for cid in fused]
    ranked = reranker.rerank(question, cands)
    info = {"shortlist": len(cands), "bm25_hits": len(bm_ids),
            "rerank_scores": [round(float(c.get("noul", 0.0)), 4) for c in ranked[: config.KEEP]]}
    return [by_id[c["id"]] for c in ranked], info
