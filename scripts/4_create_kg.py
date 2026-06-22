import hashlib
import re
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import chat
from pydantic import BaseModel, Field
from pyprojroot import here


def load_ontology(path: str | Path = here("data/ontology.yaml")) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def format_nodes(nodes: list[dict[str, str]]) -> str:
    return "\n".join(f"- {node['type']}: {node['description']}" for node in nodes)


def format_relationships(relationships: list[dict[str, list[str]]]) -> str:
    lines: list[str] = []
    for rel in relationships:
        source = ", ".join(rel["source"])
        target = ", ".join(rel["target"])
        lines.append(
            "- Relationship: "
            f"{rel['type']}\n"
            f"\t- Description: {rel['description']}\n"
            f"\t- Source types: {source}\n"
            f"\t- Target types: {target}"
        )
    return "\n".join(lines)


def build_prompt(ontology: dict) -> PromptTemplate:
    template = """
    You are an information extraction system.

    Extract a knowledge graph from the text.

    Return only valid JSON with the exact structure:
    {{
      "nodes": [
        {{"id": "...", "name": "...", "type": "...", "evidence": "..."}}
      ],
      "relationships": [
        {{"source": "...", "target": "...", "type": "...", "evidence": "..."}}
      ]
    }}

    ## Allowed Node Types

    {node_types}

    ## Allowed Relationships

    {relationships}

    ## Rules

    - Extract only named entities explicitly present in the text.
    - Node.name must be an exact text span from the input.
    - Evidence must be an exact quote from the text that supports the node or relationship.
    - Do not use external knowledge.
    - Do not resolve pronouns.
    - Do not create generic entities.
    - Do not create entities from titles, descriptions, roles, or nouns.
        - Examples: "the hero" -> ignore, "the goddess" -> ignore, "his men" -> ignore, "she" -> ignore
    - Do not invent aliases, synonyms, or normalized forms.
        - Example: if text says "Jove", do not output "Zeus" unless the exact text contains "Zeus".
    - Relationship type MUST be exactly one of the allowed relationship types.
    - Never create new relationship types.
    - Relationship evidence must clearly mention the relation between source and target.
    - If no allowed relationship exists, omit it.
    - If the passage contains no valid entities or relationships, return nodes: [] and relationships: [].

    ## Text

    {input_text}

    Return only JSON.
    """

    return PromptTemplate(
        input_variables=["input_text"],
        partial_variables={
            "node_types": format_nodes(ontology["nodes"]),
            "relationships": format_relationships(ontology["relationships"]),
        },
        template=template,
    )


def extract_number_from_file_name(name: str) -> int:
    """Extract the book number from filename like 'book_i (1).txt'."""
    result = re.search(r"\(\d+\)", name)
    if result is None:
        raise ValueError(f"Could not extract number from filename: {name}")

    return int(result.group()[1:-1])


def load_documents() -> list[Document]:
    """Load all book documents from data/books/."""
    books_dir = here("data/books")
    docs: list[Document] = []

    for file in books_dir.glob("*.txt"):
        book_num = extract_number_from_file_name(file.name)

        with open(file, "r") as fp:
            page_content = fp.read()

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "document_id": hashlib.sha256(
                        page_content.encode("utf-8")
                    ).hexdigest(),
                    "source": str(file),
                    "book": page_content.splitlines()[0],
                    "book_num": book_num,
                },
            )
        )

    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " "],  # paragraph first
    )

    splits = splitter.split_documents(docs)
    print(f"Created {len(splits)} splits.")
    return splits


class Node(BaseModel):
    id: str = Field(
        description="Unique identifier for this entity. Use a simple string that is consistent across relationships."
    )
    name: str = Field(description="The entity name exactly as it appears in the text.")
    type: str = Field(
        description="The entity category. Must be one of the allowed node types from the ontology."
    )
    evidence: str = Field(description="The exact text snippet that supports this node.")


class Relationship(BaseModel):
    source: str = Field(
        description="Must exactly match a node id from the nodes list. Never include explanations."
    )
    target: str = Field(
        description="Must exactly match a node id from the nodes list. Never include explanations."
    )
    type: str = Field(
        description="The relationship type. Must be one of the allowed relationship types from the ontology."
    )
    evidence: str = Field(
        description="The exact text snippet that supports this relationship."
    )


class ExtractionResult(BaseModel):
    nodes: list[Node] = Field(description="All entities extracted from the text.")
    relationships: list[Relationship] = Field(
        description="All relationships between extracted entities."
    )


def extract_structure(
    documents: list[Document], prompt: PromptTemplate
) -> tuple[list[Node], list[Relationship]]:
    nodes = []
    relationships = []
    for document in documents:
        response = chat(
            model="llama3.1:8b",
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(input_text=document.page_content),
                }
            ],
            format=ExtractionResult.model_json_schema(),
        )

        if response.message.content is None:
            raise RuntimeError("Ollama returned no content")

        result = ExtractionResult.model_validate_json(response.message.content)

        nodes.extend(result.nodes)
        relationships.extend(result.relationships)

    return nodes, relationships


def main() -> None:

    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    print("\nSplitting documents...")
    splits = split_documents(docs)

    ontology = load_ontology()
    prompt = build_prompt(ontology)

    nodes, relations = extract_structure(splits, prompt)
    print(
        f"Extracted {len(nodes)} nodes and {len(relations)} relations from {len(splits)} documents."
    )
    print(f"\n{'-'*20}\n{nodes=}\n{relations=}")


if __name__ == "__main__":
    main()
