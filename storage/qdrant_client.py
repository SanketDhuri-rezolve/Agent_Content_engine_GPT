"""Thin wrapper interface around Qdrant. Step 1 only needs upsert/search to
exist and be swappable/mockable — no real vectors flow until Step 2 wires in
InternVideo2 embeddings."""

from abc import ABC, abstractmethod
from functools import lru_cache

from config import get_settings


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, collection: str, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None: ...

    @abstractmethod
    def search(self, collection: str, query_vector: list[float], top_k: int) -> list[dict]:
        """Returns a list of {id, score, payload} dicts."""

    @abstractmethod
    def ensure_collection(self, collection: str, vector_size: int) -> None: ...


class QdrantVectorStore(VectorStore):
    def __init__(self, url: str, api_key: str | None):
        from qdrant_client import QdrantClient

        self._client = QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        existing = [c.name for c in self._client.get_collections().collections]
        if collection not in existing:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert(self, collection: str, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=id_, vector=vector, payload=payload)
            for id_, vector, payload in zip(ids, vectors, payloads)
        ]
        self._client.upsert(collection_name=collection, points=points)

    def search(self, collection: str, query_vector: list[float], top_k: int) -> list[dict]:
        results = self._client.search(collection_name=collection, query_vector=query_vector, limit=top_k)
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]


class InMemoryVectorStore(VectorStore):
    """Zero-dependency stand-in for local dev/tests (Step 1) — avoids
    requiring a running Qdrant instance just to exercise pipeline logic."""

    def __init__(self):
        self._collections: dict[str, dict[str, tuple[list[float], dict]]] = {}

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        self._collections.setdefault(collection, {})

    def upsert(self, collection: str, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        store = self._collections.setdefault(collection, {})
        for id_, vector, payload in zip(ids, vectors, payloads):
            store[id_] = (vector, payload)

    def search(self, collection: str, query_vector: list[float], top_k: int) -> list[dict]:
        import math

        store = self._collections.get(collection, {})

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
            norm_b = math.sqrt(sum(x * x for x in b)) or 1e-9
            return dot / (norm_a * norm_b)

        scored = [
            {"id": id_, "score": cosine(query_vector, vector), "payload": payload}
            for id_, (vector, payload) in store.items()
        ]
        return sorted(scored, key=lambda r: r["score"], reverse=True)[:top_k]


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    if settings.environment == "local":
        return InMemoryVectorStore()
    return QdrantVectorStore(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
