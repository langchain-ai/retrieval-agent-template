"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and Weaviate.

Functions:
    make_elastic_retriever: Context manager for creating an Elasticsearch retriever.
    make_pinecone_retriever: Context manager for creating a Pinecone retriever.
    make_retriever: Factory function to create a retriever based on configuration.

The retrievers support filtering results by user_id to ensure data isolation between users.
"""

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings

from retrieval_graph.configuration import Configuration, IndexConfiguration

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStoreRetriever


@contextmanager
def make_elastic_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    from langchain_elasticsearch import ElasticsearchStore

    vstore = ElasticsearchStore(
        es_url=os.environ["ELASTICSEARCH_URL"],
        es_api_key=os.environ["ELASTICSEARCH_API_KEY"],
        index_name="langchain_index",
        embedding=embedding_model,
    )

    search_kwargs = configuration.search_kwargs

    search_filter = search_kwargs.setdefault("filter", [])
    search_filter.append({"term": {"metadata.user_id": configuration.user_id}})
    yield vstore.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_pinecone_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific pinecone index."""
    from langchain_pinecone import PineconeVectorStore

    search_kwargs = configuration.search_kwargs

    search_filter = search_kwargs.setdefault("filter", {})
    search_filter.update({"user_id": configuration.user_id})
    vstore = PineconeVectorStore.from_existing_index(
        os.environ["PINECONE_INDEX_NAME"], embedding=embedding_model
    )
    yield vstore.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_weaviate_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific weaviate index."""
    import weaviate
    from langchain_weaviate import WeaviateVectorStore
    from weaviate.classes.query import Filter

    search_kwargs = configuration.search_kwargs
    search_filters = search_kwargs.setdefault("filters", [])
    search_filters.extend(Filter.by_property("user_id").equal(configuration.user_id))
    weaviate_client = weaviate.connect_to_wcs(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=weaviate.classes.init.Auth.api_key(
            os.environ["WEAVIATE_API_KEY"]
        ),
        skip_init_checks=True,
    )
    vstore = WeaviateVectorStore(
        client=weaviate_client,
        index_name=os.environ["WEAVIATE_INDEX_NAME"],
        text_key="text",
        embedding=embedding_model,
        attributes=["source", "title"],
    )
    yield vstore.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_mongodb_retriver(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

    vstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        namespace=f"langgraph_retrieval_agent.{configuration.user_id}",
        embedding=embedding_model,
    )
    yield vstore.as_retriever()


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = IndexConfiguration.from_runnable_config(config)
    embedding_model = OpenAIEmbeddings(model=configuration.embedding_model_name)
    user_id = configuration.user_id
    if not user_id:
        raise ValueError("Please provide a valid user_id in the configuration.")
    match configuration.retriever_provider:
        case "elastic":
            with make_elastic_retriever(configuration, embedding_model) as retriever:
                yield retriever
        case "pinecone":
            with make_pinecone_retriever(configuration, embedding_model) as retriever:
                yield retriever
        case "weaviate":
            with make_weaviate_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "mongodb":
            with make_mongodb_retriver(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(Configuration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )
