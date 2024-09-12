"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config


@dataclass(kw_only=True)
class IndexConfiguration:
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including user identification, embedding model selection,
    retriever provider choice, and search parameters.
    """

    user_id: str
    """Unique identifier for the user."""

    embedding_model: Annotated[
        str,
        # This metadata is only used for the template registry.
        # You may remove in your own code
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = "openai/text-embedding-3-small"
    """Name of the embedding model to use. Must be a valid embedding model name."""

    retriever_provider: Annotated[
        Literal["elastic", "pinecone", "mongodb"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = "elastic"
    """The vector store provider to use for retrieval. Options are 'elastic', 'pinecone', or 'mongodb'."""

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

    thread_id: str
    response_system_prompt: str = """You are a helpful AI assistant. Answer the user's questions based on the retrieved documents.

{retrieved_docs}"

System time: {system_time}"""
    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = (
        "anthropic/claude-3-5-sonnet-20240620"
    )
    query_system_prompt: str = """Generate search queries to retrieve documents that may help answer the user's question. Previously, you made the following queries:
    
<previous_queries/>
{queries}
</previous_queries>

System time: {system_time}"""
    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = (
        "openai/gpt-4o-mini"
    )
