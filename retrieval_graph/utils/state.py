import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
from langgraph.channels.context import Context
from langgraph.graph import add_messages
from typing_extensions import Annotated, TypedDict

from retrieval_graph.utils.configuration import ensure_configurable

############################  Doc Indexing State  #############################


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


@contextmanager
def make_retriever(config: RunnableConfig):
    """Create a retriever for the agent, based on the current configuration."""
    # This is a local example.
    configuration = ensure_configurable(config)
    embedding_model = OpenAIEmbeddings(model=configuration["embedding_model_name"])
    user_id = config["configurable"]["user_id"]
    if not user_id:
        raise ValueError("Please provide a valid user_id in the configuration.")
    data_path = Path("data", user_id, "retriever.json")
    data_path.parent.mkdir(parents=True, exist_ok=True)

    connection_options = {}
    if os.environ.get("ELASTICSEARCH_API_KEY") and os.environ.get("ELASTICSEARCH_URL"):
        connection_options = {"es_api_key": os.environ["ELASTICSEARCH_API_KEY"]}
    elif os.environ.get("ELASTICSEARCH_USER") and os.environ.get(
        "ELASTICSEARCH_PASSWORD"
    ):
        connection_options = {
            "es_user": os.environ["ELASTICSEARCH_USER"],
            "es_password": os.environ["ELASTICSEARCH_PASSWORD"],
        }
    else:
        raise ValueError(
            "Please provide a valid API key or user/password for Elasticsearch."
        )

    vstore = ElasticsearchStore(
        **connection_options,  # type: ignore
        es_url=os.environ["ELASTICSEARCH_URL"],
        index_name="langchain_index",
        embedding=embedding_model,
    )

    search_kwargs = configuration["search_kwargs"] or {}

    search_filter = search_kwargs.setdefault("filter", [])
    search_filter.append({"term": {"metadata.user_id": user_id}})
    yield vstore.as_retriever(search_kwargs=search_kwargs)


# The index state defines the simple IO for the single-node index graph
class IndexState(TypedDict):
    docs: Annotated[Sequence[Document], reduce_docs]
    """A list of documents that the agent can index."""
    retriever: Annotated[VectorStoreRetriever, Context(make_retriever)]
    """The retriever is managed by LangGraph "in context."
    
    Context state vars are not serialized. Instead, they are re-created
    for each invocation of the graph."""


#############################  Agent State  ###################################


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


# This is the primary state of your agent, where you can store any information


def add_queries(existing: Sequence[str], new: Sequence[str]) -> Sequence[str]:
    return list(existing) + list(new)


class State(InputState):
    """The state of your graph / agent."""

    queries: Annotated[Sequence[str], add_queries]
    """A list of search queries that the agent has generated."""

    retrieved_docs: Sequence[Document]
    """Populated by the retriever. This is a list of documents that the agent can reference."""

    retriever: Annotated[VectorStoreRetriever, Context(make_retriever)]

    # Feel free to add additional attributes to your state as needed.
    # Common examples include retrieved documents, extracted entities, API connections, etc.
