import json
import os
from pathlib import Path
from typing import Any

from utils.utils import parse_string

# Top-level fields that are already consumed by the input/output/eval_info
# sections of the eval record. Excluded from the metadata passthrough so we
# don't duplicate them.
KNOWN_FIELDS: set[str] = {
    # input
    "context",
    "question",
    "json_schema",
    # output
    "ground_truth",
    "validated_output",
    "candidate_response",
    # eval_info / computed upstream
    "input_context_length",
    "difficulty_weight",
}

# Maps our modality enum to the on-disk directory name. Note the plural for
# `image` — existing response dir is `data/images_responses/`.
MODALITY_DIR = {
    "text": "text_responses",
    "image": "images_responses",
    "audio": "audio_responses",
}


def build_eval_record(
    record: dict[str, Any],
    candidate: Any,
    model_id: str,
    modality: str,
    input_tokens: int,
    output_tokens: int,
    avg_time: float,
    schema: dict | None = None,
) -> dict[str, Any]:
    """Build the nested {metadata, input, output, eval_info} record.

    `metadata` carries `record_id`, `model_id`, a resolved `difficulty`, plus
    a generic passthrough of any other top-level fields on the record that
    aren't part of the input/output/eval_info sections. This preserves
    modality-specific fields (`meeting_id`, `num_speakers`, `source_pdf`,
    `source_category`, `source_id`, etc.) automatically.
    """
    difficulty = (
        record.get("question_difficulty")
        or record.get("schema_complexity")
        or "unknown"
    )

    # Generic metadata passthrough: everything on the record that's not
    # a known input/output field, not the record_id (handled below), and
    # not the computed difficulty keys we explicitly resolve.
    reserved = KNOWN_FIELDS | {"record_id", "question_difficulty"}
    passthrough = {k: v for k, v in record.items() if k not in reserved}

    metadata: dict[str, Any] = {
        "record_id": record.get("record_id"),
        "difficulty": difficulty,
        "model_id": model_id,
    }
    metadata.update(passthrough)

    resolved_schema = schema if schema is not None else parse_string(record.get("json_schema"))
    ground_truth = record.get("ground_truth")
    if ground_truth is not None:
        ground_truth = parse_string(ground_truth)

    return {
        "metadata": metadata,
        "input": {
            "context": record.get("context", ""),
            "question": record.get("question", ""),
            "json_schema": resolved_schema,
        },
        "output": {
            "candidate_response": candidate,
            "ground_truth": ground_truth,
        },
        "eval_info": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "avg_time_per_record": avg_time,
        },
    }


def resolve_output_path(model_id: str, modality: str) -> Path:
    """Pick the right response directory and filename for `(model, modality)`.

    - Dir: `data/{text,images,audio}_responses/` locally,
      `/mnt/hf-cache/{text,images,audio}_responses/` on Modal.
    - Filename: `response_{model_slug}.jsonl` for text,
      `response_{model_slug}_{modality}.jsonl` for image/audio.
    """
    if modality not in MODALITY_DIR:
        raise ValueError(f"Unknown modality {modality!r}")

    model_slug = model_id.replace("/", "_")
    suffix = "" if modality == "text" else f"_{modality}"
    filename = f"response_{model_slug}{suffix}.jsonl"

    if os.getenv("MODAL_TASK_ID"):
        base = Path("/mnt/hf-cache") / MODALITY_DIR[modality]
    else:
        base = Path("data") / MODALITY_DIR[modality]

    return base / filename


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
