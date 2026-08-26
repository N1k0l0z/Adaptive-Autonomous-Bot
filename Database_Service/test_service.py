import pytest
import requests
import uuid

API_URL = "http://localhost:8000"


@pytest.fixture
def sample_session_id():
    return f"test_session_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sample_vector_chunk():
    return {
        "document_id": f"doc_{uuid.uuid4().hex[:6]}",
        "chunk_index": 0,
        "content": "pgvector adds vector similarity search capabilities to PostgreSQL databases.",
        "embedding": [0.05] * 1536,
        "metadata": {"source": "documentation", "topic": "database"}
    }


def test_api_health():
    res = requests.get(f"{API_URL}/docs")
    assert res.status_code == 200


def test_add_and_retrieve_conversation(sample_session_id):
    user_msg = {
        "conv_id": sample_session_id,
        "role": "user",
        "message": "Hello, how do I configure PostgreSQL with pgvector?"
    }
    res_user = requests.post(f"{API_URL}/history/add", json=user_msg)
    assert res_user.status_code in [200, 201]

    assistant_msg = {
        "conv_id": sample_session_id,
        "role": "assistant",
        "message": "You can use the official pgvector/pgvector Docker image and enable the extension."
    }
    res_assistant = requests.post(f"{API_URL}/history/add", json=assistant_msg)
    assert res_assistant.status_code in [200, 201]

    res_get = requests.get(f"{API_URL}/history/{sample_session_id}")
    assert res_get.status_code == 200

    data = res_get.json()
    messages = data["messages"] if isinstance(data, dict) else data

    assert isinstance(messages, list)
    assert len(messages) >= 2
    assert messages[0]["message"] == user_msg["message"]
    assert messages[1]["message"] == assistant_msg["message"]


def test_rag_chunk_and_vector_search(sample_vector_chunk):
    res_chunk = requests.post(f"{API_URL}/rag/chunks/add", json=sample_vector_chunk)
    assert res_chunk.status_code in [200, 201]

    search_payload = {
        "query_embedding": sample_vector_chunk["embedding"],
        "top_k": 3
    }
    res_search = requests.post(f"{API_URL}/rag/search", json=search_payload)
    assert res_search.status_code == 200

    data = res_search.json()
    results = data["results"] if isinstance(data, dict) else data

    assert isinstance(results, list)
    assert len(results) > 0
    assert any(r.get("document_id") == sample_vector_chunk["document_id"] for r in results)