"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

from typing import Sequence

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from retrieval_graph.configuration import IndexConfiguration
from retrieval_graph.state import IndexState


def ensure_docs_have_assistant_id(
    docs: Sequence[Document], config: RunnableConfig
) -> list[Document]:
    """Ensure that all documents have a assistant_id in their metadata.

        docs (Sequence[Document]): A sequence of Document objects to process.
        config (RunnableConfig): A configuration object containing the assistant_id.

    Returns:
        list[Document]: A new list of Document objects with updated metadata.
    """
    assistant_id = config["configurable"]["assistant_id"]
    return [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "assistant_id": assistant_id},
        )
        for doc in docs
    ]


async def index_docs(
    state: IndexState, *, config: RunnableConfig | None = None
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.r
    """
    if not config:
        raise ValueError("Configuration required to run index_docs.")
    stamped_docs = ensure_docs_have_assistant_id(state.docs, config)

    await state.retriever.aadd_documents(stamped_docs)
    return {"docs": "delete"}


# Define a new graph


builder = StateGraph(IndexState, config_schema=IndexConfiguration)
builder.add_node(index_docs)
builder.add_edge("__start__", "index_docs")
# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile()
