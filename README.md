---
name: retrieval-agent-template
schema: |
  config_schemas:
    indexer:
      $ref: '#/$defs/IndexConfiguration'
      $defs:
        IndexConfiguration:
          title: IndexConfiguration
          type: object
          properties:
            user_id:
              type: string
            embedding_model_name:
              __lg_studio_meta:
                kind: embedding
                options:
                  - text-embedding-3-small
                  - text-embedding-3-large
                  - text-embedding-ada-002
              type: string
              default: text-embedding-3-small
            retriever_provider:
              __lg_studio_meta:
                kind: retriever
                options:
                  - elastic
                  - pinecone
              enum:
                - elastic
                - pinecone
              default: elastic
            search_kwargs:
              type: object
          required:
            - user_id
    retrieval_graph:
      $ref: '#/$defs/Configuration'
      $defs:
        Configuration:
          title: Configuration
          description: The configuration for the agent.
          type: object
          properties:
            user_id:
              type: string
            thread_id:
              type: string
            embedding_model_name:
              __lg_studio_meta:
                kind: embedding
                options:
                  - text-embedding-3-small
                  - text-embedding-3-large
                  - text-embedding-ada-002
              type: string
              default: text-embedding-3-small
            retriever_provider:
              __lg_studio_meta:
                kind: retriever
                options:
                  - elastic
                  - pinecone
              enum:
                - elastic
                - pinecone
              default: elastic
            search_kwargs:
              type: object
            response_system_prompt:
              type: string
              default: >
                You are a helpful AI assistant. Answer the user's questions based on the retrieved documents.

                {retrieved_docs}"

                System time: {system_time}
            response_model_name:
              __lg_studio_meta:
                kind: llm
                options:
                  - gpt-4o
                  - gpt-4o-mini
                  - gpt-4
                  - gpt-4-turbo
                  - gpt-4-0314
                  - gpt-4-0613
                  - gpt-4-32k
                  - gpt-4-32k-0314
                  - gpt-4-32k-0613
                  - gpt-4-turbo-preview
                  - gpt-4-1106-preview
                  - gpt-4-0125-preview
                  - gpt-4-vision-preview
                  - gpt-3.5-turbo
                  - gpt-3.5-turbo-0301
                  - gpt-3.5-turbo-0613
                  - gpt-3.5-turbo-1106
                  - gpt-3.5-turbo-0125
                  - gpt-3.5-turbo-16k
                  - gpt-3.5-turbo-16k-0613
                  - claude-3-5-sonnet-20240620
                  - claude-3-opus-20240229
                  - claude-3-sonnet-20240229
                  - claude-3-haiku-20240307
                  - claude-instant-1.2
                  - claude-1.2
                  - claude-2.0
                  - claude-2.1
                  - mythomax-l2-13b
                  - starcoder-16b
                  - mixtral-8x7b-instruct
                  - mixtral-8x22b-instruct
                  - llama-v3-8b-instruct
                  - llama-v3-70b-instruct
                  - llama-v3p1-405b-instruct-long
                  - yi-large
                  - gemma2-9b-it
                  - mixtral-8x7b-instruct-hf
                  - phi-3-vision-128k-instruct
                  - llama-v3-70b-instruct-hf
                  - llama-v3p1-405b-instruct
                  - llama-v3-8b-instruct-hf
                  - phi-3p5-vision-instruct
                  - llama-v3p1-8b-instruct
                  - llama-v3p1-70b-instruct
              type: string
              default: claude-3-5-sonnet-20240620
            query_system_prompt:
              type: string
              default: >
                Generate search queries to retrieve documents that may help answer the user's question. Previously, you made the following queries:
                    
                <previous_queries/>
                {queries}
                </previous_queries>

                System time: {system_time}
            query_model_name:
              __lg_studio_meta:
                kind: llm
                options:
                  - gpt-4o
                  - gpt-4o-mini
                  - gpt-4
                  - gpt-4-turbo
                  - gpt-4-0314
                  - gpt-4-0613
                  - gpt-4-32k
                  - gpt-4-32k-0314
                  - gpt-4-32k-0613
                  - gpt-4-turbo-preview
                  - gpt-4-1106-preview
                  - gpt-4-0125-preview
                  - gpt-4-vision-preview
                  - gpt-3.5-turbo
                  - gpt-3.5-turbo-0301
                  - gpt-3.5-turbo-0613
                  - gpt-3.5-turbo-1106
                  - gpt-3.5-turbo-0125
                  - gpt-3.5-turbo-16k
                  - gpt-3.5-turbo-16k-0613
                  - claude-3-5-sonnet-20240620
                  - claude-3-opus-20240229
                  - claude-3-sonnet-20240229
                  - claude-3-haiku-20240307
                  - claude-instant-1.2
                  - claude-1.2
                  - claude-2.0
                  - claude-2.1
                  - mythomax-l2-13b
                  - starcoder-16b
                  - mixtral-8x7b-instruct
                  - mixtral-8x22b-instruct
                  - llama-v3-8b-instruct
                  - llama-v3-70b-instruct
                  - llama-v3p1-405b-instruct-long
                  - yi-large
                  - gemma2-9b-it
                  - mixtral-8x7b-instruct-hf
                  - phi-3-vision-128k-instruct
                  - llama-v3-70b-instruct-hf
                  - llama-v3p1-405b-instruct
                  - llama-v3-8b-instruct-hf
                  - phi-3p5-vision-instruct
                  - llama-v3p1-8b-instruct
                  - llama-v3p1-70b-instruct
              type: string
              default: gpt-4o
          required:
            - user_id
            - thread_id
---

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
