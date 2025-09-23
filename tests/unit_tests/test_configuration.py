import os
from unittest.mock import patch

from retrieval_graph.configuration import Configuration, IndexConfiguration


def test_configuration_defaults() -> None:
    """Test that Configuration can be created with default values."""
    config = Configuration()
    assert config.user_id == ""
    assert config.embedding_model == "openai/text-embedding-3-small"
    assert config.retriever_provider == "elastic"
    assert config.response_model == "anthropic/claude-3-5-sonnet-20240620"
    assert config.query_model == "anthropic/claude-3-haiku-20240307"


def test_configuration_with_values() -> None:
    """Test that Configuration can be created with explicit values."""
    config = Configuration(
        user_id="test_user",
        embedding_model="openai/text-embedding-ada-002",
        retriever_provider="pinecone",
        response_model="anthropic/claude-3-opus-20240229",
        query_model="anthropic/claude-3-sonnet-20240229",
    )
    assert config.user_id == "test_user"
    assert config.embedding_model == "openai/text-embedding-ada-002"
    assert config.retriever_provider == "pinecone"
    assert config.response_model == "anthropic/claude-3-opus-20240229"
    assert config.query_model == "anthropic/claude-3-sonnet-20240229"


def test_configuration_post_init_with_env_vars() -> None:
    """Test that __post_init__ populates fields from environment variables when using defaults."""
    with patch.dict(
        os.environ,
        {
            "USER_ID": "env_user",
            "EMBEDDING_MODEL": "cohere/embed-english-v3.0",
            "RETRIEVER_PROVIDER": "mongodb",
            "RESPONSE_MODEL": "openai/gpt-4",
            "QUERY_MODEL": "openai/gpt-3.5-turbo",
        },
    ):
        config = Configuration()
        assert config.user_id == "env_user"
        assert config.embedding_model == "cohere/embed-english-v3.0"
        assert config.retriever_provider == "mongodb"
        assert config.response_model == "openai/gpt-4"
        assert config.query_model == "openai/gpt-3.5-turbo"


def test_configuration_post_init_explicit_values_override_env() -> None:
    """Test that explicit values take precedence over environment variables."""
    with patch.dict(
        os.environ,
        {
            "USER_ID": "env_user",
            "EMBEDDING_MODEL": "cohere/embed-english-v3.0",
            "RETRIEVER_PROVIDER": "mongodb",
            "RESPONSE_MODEL": "openai/gpt-4",
            "QUERY_MODEL": "openai/gpt-3.5-turbo",
        },
    ):
        config = Configuration(
            user_id="explicit_user",
            embedding_model="openai/text-embedding-ada-002",
            retriever_provider="pinecone",
            response_model="anthropic/claude-3-opus-20240229",
            query_model="anthropic/claude-3-sonnet-20240229",
        )
        assert config.user_id == "explicit_user"
        assert config.embedding_model == "openai/text-embedding-ada-002"
        assert config.retriever_provider == "pinecone"
        assert config.response_model == "anthropic/claude-3-opus-20240229"
        assert config.query_model == "anthropic/claude-3-sonnet-20240229"


def test_index_configuration_post_init() -> None:
    """Test that IndexConfiguration __post_init__ works correctly."""
    with patch.dict(
        os.environ,
        {
            "USER_ID": "index_user",
            "EMBEDDING_MODEL": "cohere/embed-english-v3.0",
            "RETRIEVER_PROVIDER": "mongodb",
        },
    ):
        config = IndexConfiguration()
        assert config.user_id == "index_user"
        assert config.embedding_model == "cohere/embed-english-v3.0"
        assert config.retriever_provider == "mongodb"
