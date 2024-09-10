"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Callable, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

# This will live in another package I think
from msgspec import Meta
from typing_extensions import Annotated


def accept_all(_: Any) -> bool:
    """Accept any value."""
    return True


def StudioSpec(
    *,
    kind: Optional[Literal["llm", "embedding", "retriever"]],
    matcher: Optional[Callable[[Any], bool]] = None,
) -> Meta:
    """Metadata for the schema."""
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
class DefaultConfiguration:
    """This is automatically populated by langgraph."""

    thread_id: str
    graph_id: str
    assistant_id: str


@dataclass(kw_only=True)
class IndexConfiguration(DefaultConfiguration):
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including user identification, embedding model selection,
    retriever provider choice, and search parameters.
    """

    embedding_model_name: Annotated[
        str,
        StudioSpec(kind="embedding", matcher=_valid_embeddings),
    ] = "text-embedding-3-small"
    """Name of the embedding model to use. Must be a valid embedding model name."""

    retriever_provider: Annotated[
        Literal["elastic", "pinecone", "mongodb", "weaviate"],
        StudioSpec(kind="retriever"),
    ] = "elastic"
    """The vector store provider to use for retrieval. Options are 'elastic', 'pinecone', or 'weaviate'."""

    search_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to pass to the search function of the retriever."""

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of IndexConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=IndexConfiguration)


@dataclass(kw_only=True)
class Configuration(IndexConfiguration):
    """The configuration for the agent."""

    response_system_prompt: str = """You are a helpful AI assistant. Answer the user's questions based on the retrieved documents.

{retrieved_docs}"

System time: {system_time}"""
    response_model_name: Annotated[str, StudioSpec(kind="llm")] = (
        "claude-3-5-sonnet-20240620"
    )
    query_system_prompt: str = """Generate search queries to retrieve documents that may help answer the user's question. Previously, you made the following queries:
    
<previous_queries/>
{queries}
</previous_queries>

System time: {system_time}"""
    query_model_name: Annotated[str, StudioSpec(kind="llm")] = "gpt-4o"
