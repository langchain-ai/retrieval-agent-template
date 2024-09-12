<!--
Configuration auto-generated by `langgraph template lock`. DO NOT EDIT MANUALLY.
{
  "config_schemas": {
    "indexer": {
      "type": "object",
      "properties": {
        "user_id": {
          "type": "string"
        },
        "embedding_model_name": {
          "type": "string",
          "default": "text-embedding-3-small",
          "environment": [
            {
              "value": "cohere",
              "variables": "COHERE_API_KEY"
            },
            {
              "value": "openai",
              "variables": "OPENAI_API_KEY"
            }
          ]
        },
        "retriever_provider": {
          "enum": [
            "elastic",
            "mongodb",
            "pinecone",
            "weaviate"
          ],
          "default": "elastic",
          "environment": [
            {
              "value": "elastic",
              "variables": [
                "ELASTICSEARCH_URL",
                "ELASTICSEARCH_API_KEY"
              ]
            },
            {
              "value": "mongodb",
              "variables": [
                "MONGODB_URI"
              ]
            },
            {
              "value": "pinecone",
              "variables": [
                "PINECONE_API_KEY",
                "PINECONE_INDEX_NAME"
              ]
            },
            {
              "value": "weaviate",
              "variables": [
                "WEAVIATE_URL",
                "WEAVIATE_API_KEY",
                "WEAVIATE_INDEX_NAME"
              ]
            }
          ]
        },
        "search_kwargs": {
          "type": "object"
        }
      }
    },
    "retrieval_graph": {
      "type": "object",
      "properties": {
        "user_id": {
          "type": "string"
        },
        "thread_id": {
          "type": "string"
        },
        "embedding_model_name": {
          "type": "string",
          "default": "text-embedding-3-small",
          "environment": [
            {
              "value": "cohere",
              "variables": "COHERE_API_KEY"
            },
            {
              "value": "openai",
              "variables": "OPENAI_API_KEY"
            }
          ]
        },
        "retriever_provider": {
          "enum": [
            "elastic",
            "mongodb",
            "pinecone",
            "weaviate"
          ],
          "default": "elastic",
          "environment": [
            {
              "value": "elastic",
              "variables": [
                "ELASTICSEARCH_URL",
                "ELASTICSEARCH_API_KEY"
              ]
            },
            {
              "value": "mongodb",
              "variables": [
                "MONGODB_URI"
              ]
            },
            {
              "value": "pinecone",
              "variables": [
                "PINECONE_API_KEY",
                "PINECONE_INDEX_NAME"
              ]
            },
            {
              "value": "weaviate",
              "variables": [
                "WEAVIATE_URL",
                "WEAVIATE_API_KEY",
                "WEAVIATE_INDEX_NAME"
              ]
            }
          ]
        },
        "search_kwargs": {
          "type": "object"
        },
        "response_system_prompt": {
          "type": "string",
          "default": "You are a helpful AI assistant. Answer the user's questions based on the retrieved documents.\n\n{retrieved_docs}\"\n\nSystem time: {system_time}"
        },
        "response_model_name": {
          "type": "string",
          "default": "claude-3-5-sonnet-20240620",
          "environment": [
            {
              "value": "anthropic",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "fireworks",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "openai",
              "variables": "OPENAI_API_KEY"
            }
          ]
        },
        "query_system_prompt": {
          "type": "string",
          "default": "Generate search queries to retrieve documents that may help answer the user's question. Previously, you made the following queries:\n    \n<previous_queries/>\n{queries}\n</previous_queries>\n\nSystem time: {system_time}"
        },
        "query_model_name": {
          "type": "string",
          "default": "gpt-4o",
          "environment": [
            {
              "value": "anthropic",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "fireworks",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "openai",
              "variables": "OPENAI_API_KEY"
            }
          ]
        }
      }
    }
  }
}
-->
# LangGraph Retrieval Agent Template

This LangGraph template implements a simple, extensible agent that answers questions based on a retriever.

![Graph view in LangGraph studio UI](./static/studio_ui.png)

## Project Prerequisites

Assuming you've already [installed LangGraph Studio](https://github.com/langchain-ai/langgraph-studio/releases) and cloned this repo. All that's left is to create an index in Elastic (or your selected search provider) and have an API key for the configured LLM (by default Anthropic).

Copy the [.env.example](.env.example) file. We will copy the relevant environment variables there:

```bash
cp .env.example .env
```

### Create Elastic index

Follow the [instructions here](https://python.langchain.com/v0.2/docs/integrations/vectorstores/elasticsearch/#elastic-cloud).

1. Create an account and login to Elastic cloud: https://cloud.elastic.co/login.
2. Create an index.
3. Create an API key, and copy that to the URL and API key to your `.env` file created above:

```
# Important! Replace with your variables:
ELASTICSEARCH_URL=https://abcd123455.us-west2.gcp.cloud.es.io:443
ELASTICSEARCH_API_KEY=RY92dc7b9584c8400194955f1c173ca67492dc7b9584c8400194955f1c173ca674==
```

Once you've set this up, you can open this template in LangGraph studio.

## Graphs

This template contains two graphs, each in separate files.

1. [index](./retrieval_graph/index_graph.py)

This "graph" is merely a single endpoint that lets a user post content to be indexed.

All documents are stored with the user's user_id (provided via configuration) as metadata so it can be filtered per-user.

2. [retrieval_graph](./retrieval_graph/graph.py)

This is a simple pipe for conducting conversational RAG. It has 3 primary steps as nodes.

1. Generate query: either take the user's input (if first step) or call an LLM to generate the query based on the conversation.
2. Retrieve docs: this uses the retriever [stored in context](https://langchain-ai.github.io/langgraph/how-tos/state-context-key/) to fetch docs, filtered by the user_id.
3. Generate response: call an LLM with the conversation + formatted doc results.

## Repo Structure

```txt
├── LICENSE
├── README.md
├── langgraph.json
├── poetry.lock
├── pyproject.toml
├── retrieval_graph
│   ├── __init__.py
│   ├── index_graph.py # Simple graph that exposes an api from which users can index docs
│   ├── graph.py
│   └── utils
│       ├── __init__.py
│       ├── configuration.py # Define the configurable variables
│       ├── state.py # Define state variables and how they're updated
│       └── utils.py # Other sundry utilities
└── tests # Add whatever tests you'd like here
    ├── integration_tests
    │   └── __init__.py
    └── unit_tests
        └── __init__.py
```

r
