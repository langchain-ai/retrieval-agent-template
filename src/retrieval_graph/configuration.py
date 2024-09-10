"""Define the configurable parameters for the agent."""

from typing import Any, Callable, Literal, Optional
from typing_extensions import Annotated

from langchain_core.runnables import RunnableConfig, ensure_config
from dataclasses import dataclass, field, fields


# This will live in another package I think
from msgspec import Meta


def accept_all(_):
    return True


def StudioSpec(
    *,
    kind: Optional[Literal["llm", "embedding", "retriever"]],
    matcher: Optional[Callable[[Any], bool]] = None,
):
    extra_schema = {"__lg_studio_meta": {"kind": kind}}
    extra = {"matcher": matcher or accept_all}
    return Meta(extra_json_schema=extra_schema, extra=extra)


# Below would live in the template repo.


def _valid_embeddings(name: str | Any) -> bool:
    if not isinstance(name, str):
        return False
    if name.startswith("text-embedding"):
        return True
    return False


@dataclass(kw_only=True)
class IndexConfiguration:
    user_id: str
    embedding_model_name: Annotated[
        str,
        StudioSpec(kind="embedding", matcher=_valid_embeddings),
    ] = "text-embedding-3-small"
    retriever_provider: Annotated[
        Literal["elastic", "pinecone", "weaviate"],
        StudioSpec(kind="retriever"),
    ] = "elastic"
    search_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None):
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


@dataclass(kw_only=True)
class Configuration(IndexConfiguration):
    """The configuration for the agent."""

    thread_id: str
    response_system_prompt: str = """You are a helpful AI assistant. Answer the user's questions based on the retrieved documents.

{retrieved_docs}"

System time: {system_time}"""
    response_model_name: Annotated[
        str, StudioSpec(kind="llm")
    ] = "claude-3-5-sonnet-20240620"
    query_system_prompt: str = """Generate search queries to retrieve documents that may help answer the user's question. Previously, you made the following queries:
    
<previous_queries/>
{queries}
</previous_queries>

System time: {system_time}"""
    query_model_name: Annotated[str, StudioSpec(kind="llm")] = "gpt-4o"
