"""Define a custom graph that implements a simple Reasoning and Action agent pattern.

Works with a chat model that utilizes tool calling."""

from datetime import datetime, timezone
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.pydantic_v1 import BaseModel
from langgraph.graph import StateGraph
from retrieval_graph.utils.configuration import Configuration, ensure_configurable
from retrieval_graph.utils.state import InputState, State
from retrieval_graph.utils.utils import format_docs, get_message_text

# Define the function that calls the model


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


async def index_docs(state: State):
    docs = state["docs"]
    retriever = state["retriever"]
    await retriever.aadd_documents(docs)
    return {"docs": "delete"}


async def generate_query(state: State, *, config: RunnableConfig | None = None):
    messages = state["messages"]
    if len(messages) == 1:
        # It's the first user question. We will use the input directly to search.
        human_input = get_message_text(messages[-1])
        return {"queries": [human_input]}
    else:
        configuration = ensure_configurable(config)
        # Feel free to customize the prompt, model, and other logic!
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration["query_system_prompt"]),
                ("placeholder", "{messages}"),
            ]
        )
        model = init_chat_model(
            configuration["response_model_name"]
        ).with_structured_output(SearchQuery)

        message_value = await prompt.ainvoke(
            {
                **state,
                "queries": "\n- ".join(state.get("queries") or []),
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
            config,
        )
        generated: SearchQuery = await model.ainvoke(message_value, config)
        return {
            "queries": [generated.query],
        }


async def retrieve(state: State, *, config: RunnableConfig | None = None):
    query = state["queries"][-1]
    retriever = state["retriever"]
    response = await retriever.ainvoke(query, config)
    return {"retrieved_docs": response}


async def respond(state: State, *, config: RunnableConfig | None = None):
    """Call the LLM powering our "agent"."""
    configuration = ensure_configurable(config)
    # Feel free to customize the prompt, model, and other logic!
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration["response_system_prompt"]),
            ("placeholder", "{messages}"),
        ]
    )
    model = init_chat_model(configuration["response_model_name"])

    retrieved_docs = format_docs(state["retrieved_docs"])
    message_value = await prompt.ainvoke(
        {
            **state,
            "retrieved_docs": retrieved_docs,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response: AIMessage = await model.ainvoke(message_value, config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph


builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
builder.add_node(index_docs)
builder.add_node(generate_query)
builder.add_node(retrieve)
builder.add_node(respond)


# Set the entrypoint as `respond`
# This means that this node is the first one called
def index_or_generate_query(state: State) -> Literal["index_docs", "generate_query"]:
    if state["docs"]:
        return "index_docs"
    else:
        return "generate_query"


builder.add_conditional_edges("__start__", index_or_generate_query)


def also_generate_query(state: State) -> Literal["generate_query", "__end__"]:
    if state["messages"] and state["messages"][-1].type == "human":
        return "generate_query"
    return "__end__"


builder.add_conditional_edges("index_docs", also_generate_query)
builder.add_edge("generate_query", "retrieve")
builder.add_edge("retrieve", "respond")
builder.add_edge("respond", "__end__")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
