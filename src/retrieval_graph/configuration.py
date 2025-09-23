"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from retrieval_graph import prompts


@dataclass(kw_only=True)
class IndexConfiguration:
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including user identification, embedding model selection,
    retriever provider choice, and search parameters.
    """

    user_id: str = field(
        default="", metadata={"description": "Unique identifier for the user."}
    )

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="openai/text-embedding-3-small",
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    retriever_provider: Annotated[
        Literal["elastic", "elastic-local", "pinecone", "mongodb"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="elastic",
        metadata={
            "description": "The vector store provider to use for retrieval. Options are 'elastic', 'pinecone', or 'mongodb'."
        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Additional keyword arguments to pass to the search function of the retriever."
        },
    )

    def __post_init__(self) -> None:
        """Populate fields from environment variables if not already set."""
        # Only populate from environment variables if the field is not already set
        if not self.user_id:
            self.user_id = os.environ.get("USER_ID", "")

        if self.embedding_model == "openai/text-embedding-3-small":
            self.embedding_model = os.environ.get(
                "EMBEDDING_MODEL", "openai/text-embedding-3-small"
            )

        if self.retriever_provider == "elastic":
            self.retriever_provider = os.environ.get("RETRIEVER_PROVIDER", "elastic")  # type: ignore


@dataclass(kw_only=True)
class Configuration(IndexConfiguration):
    """The configuration for the agent."""

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating responses."},
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The language model used for generating responses. Should be in the form: provider/model-name."
        },
    )

    query_system_prompt: str = field(
        default=prompts.QUERY_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for processing and refining queries."
        },
    )

    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-haiku-20240307",
        metadata={
            "description": "The language model used for processing and refining queries. Should be in the form: provider/model-name."
        },
    )

    def __post_init__(self) -> None:
        """Populate fields from environment variables if not already set."""
        # Call parent's __post_init__ first
        super().__post_init__()

        # Only populate from environment variables if the field is using the default value
        if self.response_model == "anthropic/claude-3-5-sonnet-20240620":
            self.response_model = os.environ.get(
                "RESPONSE_MODEL", "anthropic/claude-3-5-sonnet-20240620"
            )

        if self.query_model == "anthropic/claude-3-haiku-20240307":
            self.query_model = os.environ.get(
                "QUERY_MODEL", "anthropic/claude-3-haiku-20240307"
            )
