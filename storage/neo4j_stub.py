"""Graph storage interface — STUB ONLY, not implemented.

Reserved for a future phase (e.g. character/scene relationship graphs).
`NoOpGraphStore` is the only concrete implementation; it accepts calls and
does nothing, so calling code can be written against the interface today
without a Neo4j instance existing anywhere yet."""

from abc import ABC, abstractmethod


class GraphStore(ABC):
    @abstractmethod
    def upsert_node(self, label: str, node_id: str, properties: dict) -> None: ...

    @abstractmethod
    def upsert_relationship(self, from_id: str, to_id: str, rel_type: str, properties: dict) -> None: ...

    @abstractmethod
    def query(self, cypher: str, params: dict) -> list[dict]: ...


class NoOpGraphStore(GraphStore):
    def upsert_node(self, label: str, node_id: str, properties: dict) -> None:
        return None

    def upsert_relationship(self, from_id: str, to_id: str, rel_type: str, properties: dict) -> None:
        return None

    def query(self, cypher: str, params: dict) -> list[dict]:
        return []


def get_graph_store() -> GraphStore:
    return NoOpGraphStore()
