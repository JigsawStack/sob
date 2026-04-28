import copy
from typing import Any

from utils.utils import parse_string, extract_json

parse_if_string = parse_string
__all__ = [
    "parse_if_string",
    "extract_json",
    "normalize_schema_strict",
    "sanitize_schema_for_gemini",
    "ALLOWED_SCHEMA_KEYS",
]

ALLOWED_SCHEMA_KEYS = {
    "type",
    "properties",
    "items",
    "required",
    "enum",
    "description",
    "nullable",
    "format",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
}


def normalize_schema_strict(schema: Any) -> Any:
    """Enforce OpenAI strict-mode invariants on a schema.

    - Every object's `required` = list of its property keys.
    - Every object gets `additionalProperties = False`.
    - Recurses into array items.

    Returns a deepcopy; does NOT mutate the input.
    """
    out = copy.deepcopy(schema)
    _normalize_inplace(out)
    return out


def _normalize_inplace(schema: Any) -> None:
    if not isinstance(schema, dict):
        return

    if schema.get("type") == "object" and "properties" in schema:
        props = schema["properties"]
        if isinstance(props, dict):
            schema["required"] = list(props.keys())
            schema["additionalProperties"] = False
            for v in props.values():
                _normalize_inplace(v)

    if schema.get("type") == "array" and "items" in schema:
        _normalize_inplace(schema["items"])


def sanitize_schema_for_gemini(schema: Any) -> Any:
    """Strip unsupported keys + lowercase types for google-genai `response_schema`.

    - Recursively keeps only ALLOWED_SCHEMA_KEYS.
    - Lowercases `type` strings (Gemini rejects "STRING").
    - Prunes `required` to only keys that survive filtering in `properties`.

    Returns a deepcopy; does NOT mutate the input.
    """
    return _sanitize(copy.deepcopy(schema), is_properties=False)


def _sanitize(schema: Any, is_properties: bool) -> Any:
    if isinstance(schema, dict):
        if is_properties:
            # This dict is the "properties" map itself — its keys are property
            # names, not schema keywords, so don't filter them.
            return {k: _sanitize(v, is_properties=False) for k, v in schema.items()}

        out: dict[str, Any] = {}
        for k, v in schema.items():
            if k not in ALLOWED_SCHEMA_KEYS:
                continue
            if k == "type" and isinstance(v, str):
                out[k] = v.lower()
            elif k == "properties":
                out[k] = _sanitize(v, is_properties=True)
            else:
                out[k] = _sanitize(v, is_properties=False)

        if (
            "required" in out
            and "properties" in out
            and isinstance(out["required"], list)
            and isinstance(out["properties"], dict)
        ):
            out["required"] = [r for r in out["required"] if r in out["properties"]]

        return out

    if isinstance(schema, list):
        return [_sanitize(x, is_properties=False) for x in schema]

    return schema
