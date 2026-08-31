import json
import os
from typing import Any, Dict, List

import requests

DATABASE_SERVICE_URL = os.getenv("DATABASE_SERVICE_URL", "http://db_service:8000")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding_service:8001")


def fetch_conversation_history(conv_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    if not conv_id or not DATABASE_SERVICE_URL:
        return []

    url = f"{DATABASE_SERVICE_URL}/history/{conv_id}?limit={limit}"
    try:
        response = requests.get(url, timeout=5.0)
        if response.status_code == 200:
            return response.json().get("messages", [])
        return []
    except Exception:
        return []


def log_message_to_history(conv_id: str, role: str, message_payload: Any) -> None:
    if not conv_id or not message_payload or not DATABASE_SERVICE_URL:
        return

    message_str = (
        json.dumps(message_payload)
        if isinstance(message_payload, dict)
        else str(message_payload)
    )
    url = f"{DATABASE_SERVICE_URL}/history/add"
    payload = {"conv_id": conv_id, "role": role, "message": message_str}

    try:
        requests.post(url, json=payload, timeout=5.0)
    except Exception:
        pass


def get_text_embedding(text: str) -> List[float]:
    if not EMBEDDING_SERVICE_URL:
        return []

    url = f"{EMBEDDING_SERVICE_URL}/embed"
    try:
        res = requests.post(url, json={"text": text}, timeout=5.0)
        res.raise_for_status()
        return res.json().get("embeddings", [[]])[0]
    except Exception:
        return []


def perform_vector_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        embed_res = requests.post(
            f"{EMBEDDING_SERVICE_URL}/embed",
            json={"text": query},
            timeout=10,
        )
        embed_res.raise_for_status()
        res_data = embed_res.json()

        query_embedding = None
        if isinstance(res_data, dict):
            raw_emb = res_data.get("embeddings") or res_data.get("embedding")
            if isinstance(raw_emb, list) and len(raw_emb) > 0:
                query_embedding = (
                    raw_emb[0] if isinstance(raw_emb[0], list) else raw_emb
                )

        if not query_embedding:
            return []

        payload = {"query_embedding": query_embedding, "top_k": top_k}
        search_res = requests.post(
            f"{DATABASE_SERVICE_URL}/rag/search",
            json=payload,
            timeout=10,
        )
        search_res.raise_for_status()
        return search_res.json()

    except Exception:
        return []