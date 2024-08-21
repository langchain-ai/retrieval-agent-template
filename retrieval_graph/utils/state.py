from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Optional, Sequence, Union
import uuid

from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import add_messages
from langgraph.channels.context import Context
from retrieval_graph.utils.configuration import ensure_configurable
from typing_extensions import Annotated, TypedDict


def reduce_docs(
    existing: Optional[Sequence[Document]],
    new: Union[
        Sequence[Document], Sequence[dict], Sequence[str], str, Literal["delete"]
    ],
) -> Sequence[Document]:
    if new == "delete":
        return []
    if isinstance(new, str):
        return [Document(page_content=new, metadata={"id": str(uuid.uuid4())})]
    if isinstance(new, list):
        coerced = []
        for item in new:
            if isinstance(item, str):
                coerced.append(
                    Document(page_content=item, metadata={"id": str(uuid.uuid4())})
                )
            elif isinstance(item, dict):
                coerced.append(Document(**item))
            else:
                coerced.append(item)
        return coerced
    return existing or []


# Optional, the InputState is a restricted version of the State that is used to
# define a narrower interface to the outside world vs. what is maintained
# internally.
class InputState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    """Messages track the primary execution state of the agent.

    Typically accumulates a pattern of Human/AI/Human/AI messages; if
    you were to combine this template with a tool-calling ReAct agent pattern,
    it may look like this:

    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
         information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    
        (... repeat steps 2 and 3 as needed ...)
    4. AIMessage without .tool_calls - agent responding in unstructured
        format to the user.

    5. HumanMessage - user responds with the next conversational turn.

        (... repeat steps 2-5 as needed ... )
    
    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`."""
    docs: Annotated[Sequence[Document], reduce_docs]
    """A list of documents that the agent can index."""


# This is the primary state of your agent, where you can store any information


def add_queries(existing: Sequence[str], new: Sequence[str]) -> Sequence[str]:
    return existing + new


@contextmanager
def make_retriever(config: RunnableConfig):
    """Create a retriever for the agent, based on the current configuration."""
    # This is a local example.
    configuration = ensure_configurable(config)
    embedding_model = OpenAIEmbeddings(model=configuration["embedding_model_name"])
    thread_id = config["configurable"]["thread_id"]
    data_path = Path("data", thread_id, "retriever.json")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    persist_path = str(data_path.absolute())
    # Hack. bad initial state. # TODO: use elastic or something
    vector_store = SKLearnVectorStore.from_texts(
        [""] * 4, embedding_model, persist_path=persist_path
    )

    try:
        yield vector_store.as_retriever(search_kwargs=configuration["search_kwargs"])
    finally:
        if vector_store._embeddings:
            vector_store.persist()


class State(InputState):
    """The state of your graph / agent."""

    queries: Annotated[Sequence[str], add_queries]
    """A list of search queries that the agent has generated."""

    retrieved_docs: Sequence[Document]
    """Populated by the retriever. This is a list of documents that the agent can reference."""

    retriever: Annotated[VectorStoreRetriever, Context(make_retriever)]

    # Feel free to add additional attributes to your state as needed.
    # Common examples include retrieved documents, extracted entities, API connections, etc.
