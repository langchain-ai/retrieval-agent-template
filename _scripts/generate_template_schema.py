"""Generate the schema."""

# This would belong in the langgraph-api actually
import importlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Literal, get_args, get_origin

import msgspec
from langgraph.graph.state import CompiledGraph

_LLMS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-4-turbo-preview",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-vision-preview",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    # Anthropic
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-instant-1.2",
    "claude-1.2",
    "claude-2.0",
    "claude-2.1",
    # Fireworks
    "mythomax-l2-13b",
    "starcoder-16b",
    "mixtral-8x7b-instruct",
    "mixtral-8x22b-instruct",
    "llama-v3-8b-instruct",
    "llama-v3-70b-instruct",
    "llama-v3p1-405b-instruct-long",
    "yi-large",
    "gemma2-9b-it",
    "mixtral-8x7b-instruct-hf",
    "phi-3-vision-128k-instruct",
    "llama-v3-70b-instruct-hf",
    "llama-v3p1-405b-instruct",
    "llama-v3-8b-instruct-hf",
    "phi-3p5-vision-instruct",
    "llama-v3p1-8b-instruct",
    "llama-v3p1-70b-instruct",
]


_EMBEDDINGS = [
    # OpenAI
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
    # Cohere
    "embed-english-v3.0",
    "embed-english-light-v3.0",
    "embed-english-v2.0",
    "embed-english-light-v2.0",
    "embed-multilingual-v3.0",
    "embed-multilingual-light-v3.0",
    "embed-multilingual-v2.0",
]

_RETRIEVERS = ["elastic", "pinecone", "chroma"]

_CANDIDATES = {"llm": _LLMS, "retriever": _RETRIEVERS, "embedding": _EMBEDDINGS}


def get_enhanced_schema(graph: CompiledGraph) -> dict:
    """Get an enhanced schema for the graph configuration.

    This function takes a CompiledGraph object and returns an enhanced JSON schema
    for its configuration. The enhancement includes adding options for fields
    with specific kinds (llm, retriever, embedding) based on predefined candidates
    and custom matchers.

    Args:
        graph (CompiledGraph): The compiled graph object to generate the schema for.

    Returns:
        dict: An enhanced JSON schema for the graph configuration.

    Raises:
        ValueError: If an unsupported studio schema kind is encountered.
    """
    config_schema: dataclass = graph.config_type
    fields_ = fields(config_schema)
    schema_name = config_schema.__name__
    # Don't support dereferencing for now.
    # Only support the first-level values
    schema = msgspec.json.schema(config_schema)
    properties = schema["$defs"][schema_name]["properties"]
    for property, spec in properties.items():
        if "__lg_studio_meta" in spec:
            meta = spec["__lg_studio_meta"]
            kind = meta["kind"]
            field_ = next(f for f in fields_ if f.name == property)
            annotations = get_args(field_.type)
            annotated_meta = next(
                ann
                for ann in annotations
                if isinstance(ann, msgspec.Meta)
                and "__lg_studio_meta" in ann.extra_json_schema
            )
            if get_origin(annotations[0]) is Literal:
                meta["options"] = list(get_args(annotations[0]))
            else:
                matcher: Callable[[str], bool] = annotated_meta.extra["matcher"]
                if kind not in _CANDIDATES:
                    raise ValueError(
                        f"Unsupported studio schema kind {kind}. Expected one of {list(_CANDIDATES)}"
                    )
                candidates = _CANDIDATES[kind]
                meta["options"] = [opt for opt in candidates if matcher(opt)]
        # else:
        # print(f"NO SPEC: {spec}")

    return schema


def import_spec(path: str, root_path: Path) -> Any:
    """Load and return an object from a specified Python file.

    This function imports a Python module from a given file path and returns a specific object from that module.

    Args:
        path (str): A string in the format "path/to/file.py:object_variable_name",
                    where the part before the colon is the relative path to the Python file,
                    and the part after the colon is the name of the object to be imported from that file.
        root_path (Path): The root directory path to use as a reference for the relative module path.

    Returns:
        Any: The imported object from the specified module.

    Raises:
        ImportError: If the module or object cannot be imported.
        AttributeError: If the specified object does not exist in the module.

    Example:
        >>> import_spec("models/my_model.py:MyClass", Path("/project"))
        <class 'models.my_model.MyClass'>
    """
    module_path, object_name = path.split(":")
    config_dir = root_path.parent
    full_module_path = config_dir / module_path
    spec = importlib.util.spec_from_file_location(module_path, full_module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, object_name)


def get_schemas_for_config(langgraph_json: dict, root_path: Path) -> dict:
    """Generate enhanced schemas for each graph specified in the langgraph JSON configuration.

    This function iterates through the graphs defined in the langgraph JSON,
    imports the compiled graph object for each, and generates an enhanced schema
    for it using the get_enhanced_schema function.

    Args:
        langgraph_json (dict): A dictionary containing the langgraph configuration,
                               with a 'graphs' key mapping graph names to their paths.
        root_path (Path): The root path used for resolving relative paths in the configuration.

    Returns:
        dict: A dictionary mapping graph names to their enhanced schemas.

    Example:
        >>> config = {"graphs": {"main": "path/to/main_graph.py:graph"}}
        >>> schemas = get_schemas_for_config(config, Path("/project"))
        >>> print(schemas.keys())
        dict_keys(['main'])
    """
    results = {}
    for graph_name, path in langgraph_json["graphs"].items():
        compiled = import_spec(path, root_path)
        results[graph_name] = get_enhanced_schema(compiled)
    return results


def process_template(config_path: str):
    """Process the langgraph.json template file and generate a corresponding langgraph.template.json file.

    This function reads the langgraph.json file, extracts the configuration schemas,
    and writes them to a new langgraph.template.json file in the same directory.

    Args:
        config_path (str): The path to the langgraph.json file.

    Raises:
        AssertionError: If the input file is not named 'langgraph.json'.

    Side effects:
        - Creates or overwrites a 'langgraph.template.json' file in the same directory as the input file.
    """
    path = Path(config_path).absolute()
    assert path.name == "langgraph.json", "Input file must be named 'langgraph.json'"
    output_path = path.parent / "langgraph.template.json"

    with path.open("r") as f:
        config = json.load(f)

    schemas = get_schemas_for_config(config, path)

    with output_path.open("w") as f:
        json.dump({"config_schemas": schemas}, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    process_template(args.path)
