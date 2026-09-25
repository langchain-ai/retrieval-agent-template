"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and MongoDB.

The retrievers support filtering results by user_id to ensure data isolation between users.
"""

import os
from contextlib import contextmanager
from typing import Generator

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from retrieval_graph.configuration import Configuration, IndexConfiguration

## Encoder constructors


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        case "cohere":
            from langchain_cohere import CohereEmbeddings

            return CohereEmbeddings(model=model)  # type: ignore
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


## Retriever constructors


@contextmanager
def make_elastic_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    from langchain_elasticsearch import ElasticsearchStore

    connection_options = {}
    if configuration.retriever_provider == "elastic-local":
        connection_options = {
            "es_user": os.environ["ELASTICSEARCH_USER"],
            "es_password": os.environ["ELASTICSEARCH_PASSWORD"],
        }

    else:
        connection_options = {"es_api_key": os.environ["ELASTICSEARCH_API_KEY"]}

    vstore = ElasticsearchStore(
        **connection_options,  # type: ignore
        es_url=os.environ["ELASTICSEARCH_URL"],
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
def make_mongodb_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

    vstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        namespace="langgraph_retrieval_agent.default",
        embedding=embedding_model,
    )
    search_kwargs = configuration.search_kwargs
    pre_filter = search_kwargs.setdefault("pre_filter", {})
    pre_filter["user_id"] = {"$eq": configuration.user_id}
    yield vstore.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_pgvector_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a pgvector index."""
    import json
    from typing import Any, List, Tuple
    from langchain_postgres.vectorstores import PGVector as OverPGVector
    from langchain_core.documents import Document

    class PGVector(OverPGVector):
        """
        A custom override of the PGVector class to handle metadata deserialization issues
        when operating in async_mode. This class addresses a known issue where metadata,
        stored as byte data, is not properly converted back into a dictionary format
        during asynchronous operations.

        The override specifically ensures that all metadata, whether stored as bytes,
        strings, or other unrecognized formats, is correctly processed into a dictionary
        format suitable for use within the application. This is crucial for maintaining
        consistency and usability of metadata across asynchronous database interactions.

        Issue Reference:
        "Metadata field not properly deserialized when using async_mode=True with PGVector #124"

        Methods:
            _results_to_docs_and_scores: Converts query results from PGVector into a list
                                         of tuples, each containing a Document and its corresponding
                                         score, while ensuring metadata is correctly deserialized.
        """
        def _results_to_docs_and_scores(self, results: Any) -> List[Tuple[Document, float]]:
            """Return docs and scores from results."""
            docs = []
            for result in results:
                # Access the metadata
                metadata = result.EmbeddingStore.cmetadata

                # Process the metadata to ensure it's a dict
                if not isinstance(metadata, dict):
                    if hasattr(metadata, 'buf'):
                        # For Fragment types (e.g., asyncpg.Record)
                        metadata_bytes = metadata.buf
                        metadata_str = metadata_bytes.decode('utf-8')
                        metadata = json.loads(metadata_str)
                    elif isinstance(metadata, str):
                        # If it's a JSON string
                        metadata = json.loads(metadata)
                    else:
                        # Handle other types if necessary
                        metadata = {}

                doc = Document(
                    id=str(result.EmbeddingStore.id),
                    page_content=result.EmbeddingStore.document,
                    metadata=metadata,
                )
                score = result.distance if self.embeddings is not None else None
                docs.append((doc, score))
            return docs

    connection_string = os.environ.get("PGVECTOR_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("PGVECTOR_CONNECTION_STRING environment variable is not set.")

    collection_name = os.environ.get("PGVECTOR_COLLECTION_NAME", "langchain")

    # Initialize the PGVector vector store with async_mode=True
    vstore = PGVector(
        connection=connection_string,
        collection_name=collection_name,
        embeddings=embedding_model,
        use_jsonb=True,
        pre_delete_collection=False,  # Set to True if you want to delete existing data
        async_mode=True
    )

    search_kwargs = configuration.search_kwargs
    user_id = configuration.user_id
    metadata_filter = search_kwargs.setdefault("filter", {})
    metadata_filter["user_id"] = {"$eq": user_id}
    yield vstore.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = IndexConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    user_id = configuration.user_id
    if not user_id:
        raise ValueError("Please provide a valid user_id in the configuration.")
    match configuration.retriever_provider:
        case "elastic" | "elastic-local":
            with make_elastic_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "pinecone":
            with make_pinecone_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "mongodb":
            with make_mongodb_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "pgvector":
            with make_pgvector_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(Configuration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )