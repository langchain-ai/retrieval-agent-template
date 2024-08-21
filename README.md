# LangGraph Retrieval Agent Template

This LangGraph template implements a simple, extensible agent that answers questions based on a retriever.

![Graph view in LangGraph studio UI](./static/studio_ui.png)

## Repo Structure

```txt
├── LICENSE
├── README.md
├── langgraph.json
├── poetry.lock
├── pyproject.toml
├── retrieval_graph
│   ├── __init__.py
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
