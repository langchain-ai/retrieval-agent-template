"""Define the configurable parameters for the agent."""

from typing import Optional

from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict


class IndexConfiguration(TypedDict):
    embedding_model_name: str
    search_kwargs: dict
    user_id: str


def ensure_index_configurable(
    config: Optional[RunnableConfig] = None,
) -> IndexConfiguration:
    """Ensure the defaults are populated."""
    configurable = (config or {}).get("configurable") or {}
    return IndexConfiguration(
        user_id=configurable["user_id"],
        embedding_model_name=configurable.get(
            "embedding_model_name",
            "text-embedding-3-small",
        ),
        search_kwargs=configurable.get("search_kwargs") or {},
    )


class Configuration(IndexConfiguration):
    """The configuration for the agent."""

    response_system_prompt: str
    response_model_name: str
    query_system_prompt: str
    query_model_name: str
    thread_id: str


def ensure_configurable(config: Optional[RunnableConfig] = None) -> Configuration:
    """Ensure the defaults are populated."""
    configurable = (config or {}).get("configurable") or {}
    return Configuration(
        user_id=configurable["user_id"],
        thread_id=configurable["thread_id"],
        response_system_prompt=configurable.get(
            "response_system_prompt",
            "You are a helpful AI assistant. Answer the user's questions based on the retrieved documents."
            "\n\n{retrieved_docs}"
            "\n\nSystem time: {system_time}",
        ),
        response_model_name=configurable.get(
            "response_model_name", "claude-3-5-sonnet-20240620"
        ),
        query_system_prompt=configurable.get(
            "query_system_prompt",
            "Generate search queries to retrieve documents that may help answer the user's question."
            "\n\nPreviously, you made the following queries:<previous_queries/>"
            "\n{queries}\n<\previous_queries/>\n\nSystem time: {system_time}",
        ),
        query_model_name=configurable.get(
            "query_model_name",
            "accounts/fireworks/models/firefunction-v2",
        ),
        # In general, you could make these things nested configs and load from a file to get the
        # defaults. But for now, we'll keep it simple.
        embedding_model_name=configurable.get(
            "embedding_model_name",
            "text-embedding-3-small",
        ),
        search_kwargs=configurable.get("search_kwargs") or {},
    )
