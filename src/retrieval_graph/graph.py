"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from datetime import datetime, timezone
from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from retrieval_graph import retrieval
from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.utils import format_docs, get_message_text

# Define the function that calls the model


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


async def generate_query(
    state: State, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate a search query based on the current state and configuration.

    This function analyzes the messages in the state and generates an appropriate
    search query. For the first message, it uses the user's input directly.
    For subsequent messages, it uses a language model to generate a refined query.

    Args:
        state (State): The current state containing messages and other information.
        config (RunnableConfig | None, optional): Configuration for the query generation process.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing a list of generated queries.

    Behavior:
        - If there's only one message (first user input), it uses that as the query.
        - For subsequent messages, it uses a language model to generate a refined query.
        - The function uses the configuration to set up the prompt and model for query generation.
    """
    messages = state.messages
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
                "messages": state.messages,
                "queries": "\n- ".join(state.queries),
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
            config,
        )
        generated = cast(SearchQuery, await model.ainvoke(message_value, config))
        return {
            "queries": [generated.query],
        }


async def retrieve(
    state: State, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the retriever, and returns
    the retrieved documents.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    with retrieval.make_retriever(config) as retriever:
        response = await retriever.ainvoke(state.queries[-1], config)
        return {"retrieved_docs": response}


async def respond(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
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

    retrieved_docs = format_docs(state.retrieved_docs)
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
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
