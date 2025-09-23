"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

from typing import Sequence

from langchain_core.documents import Document
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from retrieval_graph import retrieval
from retrieval_graph.configuration import IndexConfiguration
from retrieval_graph.state import IndexState


def ensure_docs_have_user_id(
    docs: Sequence[Document], runtime: Runtime[IndexConfiguration]
) -> list[Document]:
    """Ensure that all documents have a user_id in their metadata.

        docs (Sequence[Document]): A sequence of Document objects to process.
        runtime (Runtime[IndexConfiguration]): Runtime context containing configuration.

    Returns:
        list[Document]: A new list of Document objects with updated metadata.
    """
    user_id = runtime.context.user_id
    return [
        Document(
            page_content=doc.page_content, metadata={**doc.metadata, "user_id": user_id}
        )
        for doc in docs
    ]


async def index_docs(
    state: IndexState, *, runtime: Runtime[IndexConfiguration]
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state.

    Args:
        state (IndexState): The current state containing documents and retriever.
        runtime (Runtime[IndexConfiguration]): Runtime context containing configuration.
    """
    with retrieval.make_retriever(runtime) as retriever:
        stamped_docs = ensure_docs_have_user_id(state.docs, runtime)

        await retriever.aadd_documents(stamped_docs)
    return {"docs": "delete"}


# Define a new graph


builder = StateGraph(IndexState, context_schema=IndexConfiguration)
builder.add_node(index_docs)
builder.add_edge("__start__", "index_docs")
# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile()
graph.name = "IndexGraph"
