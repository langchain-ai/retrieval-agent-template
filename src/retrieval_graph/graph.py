"""Define a custom graph that implements a simple Reasoning and Action agent pattern.

Works with a chat model that utilizes tool calling."""

from datetime import datetime, timezone
from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.utils import format_docs, get_message_text

# Define the function that calls the model


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


async def generate_query(state: State, *, config: RunnableConfig | None = None):
    messages = state["messages"]
    if len(messages) == 1:
        # It's the first user question. We will use the input directly to search.
        human_input = get_message_text(messages[-1])
        return {"queries": [human_input]}
    else:
        configuration = Configuration.from_runnable_config(config)
        # Feel free to customize the prompt, model, and other logic!
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.query_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )
        model = init_chat_model(
            configuration.response_model_name
        ).with_structured_output(SearchQuery)

        message_value = await prompt.ainvoke(
            {
                **state,
                "queries": "\n- ".join(state.get("queries") or []),
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
            config,
        )
        generated = cast(SearchQuery, await model.ainvoke(message_value, config))
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
    configuration = Configuration.from_runnable_config(config)
    # Feel free to customize the prompt, model, and other logic!
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.response_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    model = init_chat_model(configuration.response_model_name)

    retrieved_docs = format_docs(list(state["retrieved_docs"]))
    message_value = await prompt.ainvoke(
        {
            **state,
            "retrieved_docs": retrieved_docs,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph (It's just a pipe)


builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node(generate_query)
builder.add_node(retrieve)
builder.add_node(respond)
builder.add_edge("__start__", "generate_query")
builder.add_edge("generate_query", "retrieve")
builder.add_edge("retrieve", "respond")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
