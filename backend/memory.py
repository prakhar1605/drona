"""
Interview Memory — ChromaDB + Embeddings
Stores user answer embeddings to build long-term weakness profile.
Personalizes future question generation based on past performance.
"""

import os
import hashlib
from typing import List, Dict, Optional
import chromadb
from chromadb.utils import embedding_functions

_chroma_client = None
_collection = None
COLLECTION_NAME = "dronaai_memory"


def _get_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    persist_dir = os.path.join(os.path.dirname(__file__), "..", ".chroma_db")
    os.makedirs(persist_dir, exist_ok=True)

    _chroma_client = chromadb.PersistentClient(path=persist_dir)
    ef = embedding_functions.DefaultEmbeddingFunction()

    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def store_answer(
    session_id: str,
    question: str,
    user_answer: List[str],
    correct_answer: List[str],
    topic: str,
    difficulty: str,
    score_percent: float,
) -> bool:
    try:
        col = _get_collection()
        doc_text = (
            f"Topic: {topic} | Difficulty: {difficulty} | "
            f"Question: {question} | "
            f"User Answer: {'; '.join(user_answer)} | "
            f"Correct Answer: {'; '.join(correct_answer)} | "
            f"Score: {score_percent}%"
        )
        doc_id = hashlib.md5(f"{session_id}:{question}".encode()).hexdigest()

        col.upsert(
            ids=[doc_id],
            documents=[doc_text],
            metadatas=[{
                "session_id": session_id,
                "topic": topic,
                "difficulty": difficulty,
                "score_percent": score_percent,
                "correct": score_percent >= 70,
            }],
        )
        return True
    except Exception as e:
        print(f"[Memory] Store error: {e}")
        return False


def get_weak_areas(session_id: str, top_k: int = 5) -> List[str]:
    try:
        col = _get_collection()
        if col.count() == 0:
            return []

        results = col.query(
            query_texts=["weak incorrect wrong failed struggle"],
            n_results=min(top_k * 3, col.count()),
            where={"session_id": session_id},
        )
        if not results or not results["metadatas"]:
            return []

        weak: Dict[str, int] = {}
        for meta_list in results["metadatas"]:
            for meta in meta_list:
                if not meta.get("correct", True):
                    t = meta.get("topic", "General")
                    weak[t] = weak.get(t, 0) + 1

        sorted_weak = sorted(weak.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_weak[:top_k]]
    except Exception as e:
        print(f"[Memory] Query error: {e}")
        return []


def get_session_history_summary(session_id: str) -> str:
    try:
        col = _get_collection()
        if col.count() == 0:
            return ""

        results = col.get(where={"session_id": session_id})
        if not results or not results["metadatas"]:
            return ""

        metas = results["metadatas"]
        total = len(metas)
        correct = sum(1 for m in metas if m.get("correct", False))
        weak_topics = list({m["topic"] for m in metas if not m.get("correct", True)})

        return (
            f"Past performance: {correct}/{total} correct. "
            f"Weak areas from history: {', '.join(weak_topics) if weak_topics else 'none identified'}."
        )
    except Exception:
        return ""


def clear_session_memory(session_id: str) -> bool:
    try:
        col = _get_collection()
        results = col.get(where={"session_id": session_id})
        if results and results["ids"]:
            col.delete(ids=results["ids"])
        return True
    except Exception:
        return False
