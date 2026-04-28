import json
from typing import Any

from utils.utils import parse_string

SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "Return a valid JSON object that strictly follows the given JSON schema. "
    "Do not output anything except the JSON object.\n\n"
    "Rules:\n"
    "- No explanations: Do not include reasoning, analysis, or any sentences outside the JSON.\n"
    "- No markdown: Do not wrap the JSON in code blocks or add formatting like ``` or 'json'.\n"
    "- No extra text: Do not add prefixes or suffixes such as 'Answer:' or 'Output:'.\n"
    "- Follow the schema exactly: Use only the keys defined in the schema and ensure correct data types.\n"
    "- Include all required fields: Every field listed as required in the schema must appear in the JSON.\n"
    "- If unknown, return null: If the context does not contain the answer, set the field value to null instead of guessing."
)


def build_user_message(record: dict[str, Any], schema: dict | None = None) -> str:
    """Construct the user message used by every provider.

    If `schema` is passed, it is used verbatim (e.g. the provider has already
    sanitized it for its own constraints). Otherwise the record's json_schema
    is parsed and used.
    """
    if schema is None:
        schema = parse_string(record["json_schema"])
    schema_str = json.dumps(schema, indent=2)
    return (
        f"Context:\n{record['context']}\n\n"
        f"Question: {record['question']}\n\n"
        f"Respond with JSON matching this schema:\n{schema_str}"
    )
