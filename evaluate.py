#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator, ValidationError


DIFFICULTY_WEIGHTS = {
    "easy": 1.0,
    "medium": 2.0,
    "hard": 3.0,
}
WEIGHT_FIELD_PRIORITY = ("schema_complexity", "difficulty")

METRIC_DISPLAY_NAMES = {
    "json_parse_success": "JSON Parse Success",
    "json_root_structured": "Structured JSON Root",
    "schema_valid_input": "Schema Valid Input",
    "schema_compliance": "JSON Pass Rate",
    "leaf_value_em": "Truth Score",
    "value_token_f1": "Faithfulness Score",
    "hier_path_recall": "Path Recall",
    "path_set_f1": "Structure Coverage",
    "type_precision": "Type Safety",
    "required_key_recall": "Required Key Recall",
    "strict_json_em": "Perfect Response Rate",
}

CATEGORIES = {
    "Long Context Extraction": [
        "leaf_value_em",
        "value_token_f1",
        "hier_path_recall",
    ],
    "Complex Schema Handling": [
        "schema_compliance",
        "path_set_f1",
        "type_precision",
    ],
    "Multi-Context Linking": [
        "leaf_value_em",
        "value_token_f1",
    ],
    "Output Contract Reliability": [
        "json_parse_success",
        "schema_compliance",
        "type_precision",
    ],
    "Strict Precision": [
        "strict_json_em",
    ],
}

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
PUNCT_RE = re.compile(r"[^\w\s]")
INDEX_RE = re.compile(r"\[\d+\]")


@dataclass
class RecordMetrics:
    row: dict[str, Any]
    missing_gt_paths: list[str]
    missing_required_paths: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SOB response JSONL files and recreate eval_records/eval_summary outputs."
    )
    parser.add_argument(
        "input_path",
        help="A response JSONL file or a directory containing response JSONL files.",
    )
    parser.add_argument(
        "--output-root",
        default="data/evaluation",
        help="Root directory for evaluation outputs. Default: data/evaluation",
    )
    parser.add_argument(
        "--modality",
        choices=["auto", "text", "image", "audio"],
        default="auto",
        help="Response modality. Default: auto",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples for confidence intervals. Default: 1000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap sampling. Default: 42",
    )
    parser.add_argument(
        "--top-k-errors",
        type=int,
        default=20,
        help="Number of most common missing paths to include in error_analysis. Default: 20",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    if input_path.is_dir():
        response_files = sorted(input_path.glob("*.jsonl"))
    else:
        response_files = [input_path]

    if not response_files:
        raise SystemExit(f"No JSONL response files found under {input_path}")

    for response_file in response_files:
        modality = infer_modality(response_file, args.modality)
        records, data_quality = evaluate_file(response_file, modality)
        if not records:
            raise SystemExit(f"No valid records found in {response_file}")

        model_ids = sorted({record.row["model_id"] for record in records})
        output_dir = resolve_output_dir(
            Path(args.output_root), modality, response_file, model_ids
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        records_path = output_dir / "eval_records.jsonl"
        summary_path = output_dir / "eval_summary.json"

        write_records(records_path, records, modality)
        summary = build_summary(
            response_file=response_file,
            records=records,
            data_quality=data_quality,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            top_k_errors=args.top_k_errors,
            modality=modality,
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

        print(f"Wrote {records_path}")
        print(f"Wrote {summary_path}")


def infer_modality(response_file: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    path_str = response_file.as_posix().lower()
    if "text_response" in path_str:
        return "text"
    if "images_response" in path_str or "image_response" in path_str:
        return "image"
    if "audio_response" in path_str:
        return "audio"
    name = response_file.name.lower()
    if "_image" in name:
        return "image"
    if "_audio" in name:
        return "audio"
    return "text"


def resolve_output_dir(
    output_root: Path,
    modality: str,
    response_file: Path,
    model_ids: list[str],
) -> Path:
    stem = response_file.stem.removeprefix("response_")
    if modality == "audio" and stem.endswith("_audio"):
        stem = stem[: -len("_audio")]
    elif modality == "image" and stem.endswith("_image"):
        stem = stem[: -len("_image")]
    model_name = sanitize_model_id(stem)
    return output_root / modality / model_name


def sanitize_model_id(model_id: str) -> str:
    return model_id.split("/")[-1]


def write_records(
    records_path: Path, records: list[RecordMetrics], modality: str
) -> None:
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = dict(record.row)
            if modality == "image":
                payload["required_key_recall"] = payload["required_key_recall"]
                payload["missing_gt_paths"] = record.missing_gt_paths
                payload["missing_required_paths"] = record.missing_required_paths
            handle.write(json.dumps(payload) + "\n")


def evaluate_file(
    response_file: Path, modality: str
) -> tuple[list[RecordMetrics], dict[str, int]]:
    records: list[RecordMetrics] = []
    data_quality = {
        "json_parse_fail_count": 0,
        "json_non_structured_root_count": 0,
        "invalid_schema_input_count": 0,
        "unknown_difficulty_count": 0,
        "malformed_jsonl_line_count": 0,
    }

    with response_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                data_quality["malformed_jsonl_line_count"] += 1
                continue

            evaluated = evaluate_record(record, modality)
            row = evaluated.row
            if row["json_parse_success"] == 0.0:
                data_quality["json_parse_fail_count"] += 1
            if row["json_root_structured"] == 0.0:
                data_quality["json_non_structured_root_count"] += 1
            if row["schema_valid_input"] == 0.0:
                data_quality["invalid_schema_input_count"] += 1
            if row["known_difficulty"] == 0.0:
                data_quality["unknown_difficulty_count"] += 1
            records.append(evaluated)

    return records, data_quality


def evaluate_record(record: dict[str, Any], modality: str) -> RecordMetrics:
    metadata = record.get("metadata", {})
    input_block = record.get("input", {})
    output_block = record.get("output", {})

    record_id = metadata.get("record_id", "")
    model_id = metadata.get("model_id", "unknown")
    schema_complexity = metadata.get("schema_complexity") or "unknown"
    question_difficulty = metadata.get("difficulty") or "unknown"
    weighting_basis = "unknown"
    difficulty = "unknown"
    for field_name in WEIGHT_FIELD_PRIORITY:
        candidate = metadata.get(field_name)
        if candidate in DIFFICULTY_WEIGHTS:
            difficulty = candidate
            weighting_basis = field_name
            break
    difficulty_weight = DIFFICULTY_WEIGHTS.get(difficulty, 0.0)
    known_difficulty = 1.0 if difficulty in DIFFICULTY_WEIGHTS else 0.0

    schema = input_block.get("json_schema")
    candidate_raw = output_block.get("candidate_response")
    ground_truth = output_block.get("ground_truth")

    schema_valid_input = 1.0 if is_valid_schema(schema) else 0.0
    parsed_ok, candidate = parse_candidate(candidate_raw)
    root_structured = 1.0 if parsed_ok and isinstance(candidate, (dict, list)) else 0.0
    schema_compliance = (
        1.0
        if parsed_ok
        and root_structured
        and schema_valid_input
        and validates(candidate, schema)
        else 0.0
    )

    gt_leafs = flatten_leaf_paths(ground_truth)
    pred_leafs = flatten_leaf_paths(candidate) if root_structured else {}

    raw_path_recall = ratio(len(set(gt_leafs) & set(pred_leafs)), len(gt_leafs))
    raw_path_set_f1 = f1_from_counts(
        len(set(gt_leafs) & set(pred_leafs)), len(pred_leafs), len(gt_leafs)
    )
    raw_leaf_em = exact_match_ratio(gt_leafs, pred_leafs)
    raw_value_token_f1 = mean_token_f1(gt_leafs, pred_leafs)
    raw_type_precision = (
        compute_type_precision(candidate, schema)
        if root_structured and schema_valid_input
        else 0.0
    )
    required_key_recall, missing_required_paths = (
        compute_required_key_recall(candidate, schema)
        if root_structured and schema_valid_input
        else (0.0, required_paths(schema))
    )

    hardening = 1.0 if parsed_ok and root_structured and schema_compliance else 0.0
    coverage_gate = compute_coverage_gate(modality, raw_path_set_f1)

    leaf_value_em = raw_leaf_em * hardening * coverage_gate
    value_token_f1 = raw_value_token_f1 * hardening * coverage_gate
    hier_path_recall = raw_path_recall * hardening
    path_set_f1 = raw_path_set_f1 * hardening
    type_precision = raw_type_precision * hardening
    strict_json_em = (
        1.0 if canonical_json(candidate) == canonical_json(ground_truth) else 0.0
    )

    row = {
        "record_id": record_id,
        "model_id": model_id,
        "schema_complexity": schema_complexity,
        "question_difficulty": question_difficulty,
        "difficulty": difficulty,
        "weighting_basis": weighting_basis,
        "difficulty_weight": difficulty_weight,
        "known_difficulty": known_difficulty,
        "json_parse_success": 1.0 if parsed_ok else 0.0,
        "json_root_structured": root_structured,
        "schema_valid_input": schema_valid_input,
        "schema_compliance": schema_compliance,
        "leaf_value_em": leaf_value_em,
        "value_token_f1": value_token_f1,
        "hier_path_recall": hier_path_recall,
        "path_set_f1": path_set_f1,
        "type_precision": type_precision,
        "required_key_recall": required_key_recall,
        "strict_json_em": strict_json_em,
    }

    missing_gt_paths = sorted(set(gt_leafs) - set(pred_leafs))
    return RecordMetrics(
        row=row,
        missing_gt_paths=missing_gt_paths,
        missing_required_paths=missing_required_paths,
    )


def parse_candidate(candidate_raw: Any) -> tuple[bool, Any]:
    if isinstance(candidate_raw, (dict, list)):
        return True, candidate_raw
    return False, None


def extract_json_candidates(text: str) -> list[str]:
    candidates = [text]
    for match in JSON_BLOCK_RE.findall(text):
        candidates.append(match.strip())

    starts = [idx for idx, char in enumerate(text) if char in "[{"]
    for start in starts:
        for end in range(len(text), start, -1):
            segment = text[start:end].strip()
            if segment.endswith(("}", "]")):
                candidates.append(segment)
                break
    deduped: list[str] = []
    seen = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def is_valid_schema(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    try:
        Draft7Validator.check_schema(schema)
        return True
    except Exception:
        return False


def validates(candidate: Any, schema: dict[str, Any]) -> bool:
    try:
        Draft7Validator(schema).validate(candidate)
        return True
    except ValidationError:
        return False


def flatten_leaf_paths(value: Any, prefix: str = "") -> dict[str, Any]:
    leafs: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            leafs.update(flatten_leaf_paths(child, next_prefix))
        return leafs
    if isinstance(value, list):
        for idx, child in enumerate(value):
            next_prefix = f"{prefix}[{idx}]"
            leafs.update(flatten_leaf_paths(child, next_prefix))
        return leafs
    if prefix:
        leafs[prefix] = value
    return leafs


def flatten_present_paths(value: Any, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if prefix:
        paths.add(prefix)
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            paths.update(flatten_present_paths(child, next_prefix))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            next_prefix = f"{prefix}[{idx}]"
            paths.update(flatten_present_paths(child, next_prefix))
    return paths


def wildcard_path(path: str) -> str:
    return INDEX_RE.sub("[]", path)


def required_paths(schema: Any, prefix: str = "") -> list[str]:
    if not isinstance(schema, dict):
        return []

    schema_type = schema.get("type")
    results: list[str] = []

    if schema_type == "object":
        properties = schema.get("properties", {})
        for required_key in schema.get("required", []):
            child_schema = properties.get(required_key, {})
            child_prefix = f"{prefix}.{required_key}" if prefix else required_key
            results.append(child_prefix)
            results.extend(required_paths(child_schema, child_prefix))
    elif schema_type == "array":
        items_schema = schema.get("items", {})
        child_prefix = f"{prefix}[]" if prefix else "[]"
        results.extend(required_paths(items_schema, child_prefix))

    return results


def compute_required_key_recall(candidate: Any, schema: Any) -> tuple[float, list[str]]:
    req_paths = sorted(set(required_paths(schema)))
    if not req_paths:
        return 1.0, []
    present = {wildcard_path(path) for path in flatten_present_paths(candidate)}
    missing = [path for path in req_paths if path not in present]
    return ratio(len(req_paths) - len(missing), len(req_paths)), missing


def compute_type_precision(candidate: Any, schema: Any) -> float:
    pred_leafs = flatten_leaf_paths(candidate)
    if not pred_leafs:
        return 0.0

    matches = 0
    for path, value in pred_leafs.items():
        expected_type = schema_type_for_path(schema, path)
        if expected_type is not None and is_type_match(value, expected_type):
            matches += 1
    return ratio(matches, len(pred_leafs))


def schema_type_for_path(schema: Any, path: str) -> str | None:
    if not isinstance(schema, dict):
        return None
    parts = split_path(path)
    current = schema
    for part in parts:
        if isinstance(part, str):
            if current.get("type") != "object":
                return None
            current = current.get("properties", {}).get(part)
        else:
            if current.get("type") != "array":
                return None
            current = current.get("items")
        if current is None:
            return None
    return (
        normalize_schema_type(current.get("type"))
        if isinstance(current, dict)
        else None
    )


def split_path(path: str) -> list[str | int]:
    parts: list[str | int] = []
    token = ""
    idx = 0
    while idx < len(path):
        char = path[idx]
        if char == ".":
            if token:
                parts.append(token)
                token = ""
            idx += 1
            continue
        if char == "[":
            if token:
                parts.append(token)
                token = ""
            end = path.index("]", idx)
            parts.append(int(path[idx + 1 : end]))
            idx = end + 1
            continue
        token += char
        idx += 1
    if token:
        parts.append(token)
    return parts


def normalize_schema_type(schema_type: Any) -> str | None:
    if isinstance(schema_type, list):
        non_null = [item for item in schema_type if item != "null"]
        if len(non_null) == 1:
            schema_type = non_null[0]
    if schema_type in {
        "string",
        "integer",
        "number",
        "boolean",
        "object",
        "array",
        "null",
    }:
        return str(schema_type)
    return None


def json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def is_type_match(value: Any, expected_type: str) -> bool:
    actual = json_type_name(value)
    if actual == expected_type:
        return True
    # JSON Schema treats integers as valid numbers.
    if expected_type == "number" and actual == "integer":
        return True
    return False


def exact_match_ratio(gt_leafs: dict[str, Any], pred_leafs: dict[str, Any]) -> float:
    if not gt_leafs:
        return 1.0
    matches = sum(
        1
        for path, gt_val in gt_leafs.items()
        if path in pred_leafs and gt_val == pred_leafs[path]
    )
    return matches / len(gt_leafs)


def mean_token_f1(gt_leafs: dict[str, Any], pred_leafs: dict[str, Any]) -> float:
    if not gt_leafs:
        return 1.0
    return sum(
        token_f1(gt_val, pred_leafs.get(path)) for path, gt_val in gt_leafs.items()
    ) / len(gt_leafs)


def token_f1(gt_value: Any, pred_value: Any) -> float:
    gt_tokens = normalize_tokens(gt_value)
    pred_tokens = normalize_tokens(pred_value)
    if not gt_tokens and not pred_tokens:
        return 1.0
    if not gt_tokens or not pred_tokens:
        return 0.0
    gt_counter = Counter(gt_tokens)
    pred_counter = Counter(pred_tokens)
    overlap = sum((gt_counter & pred_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_counter.values())
    recall = overlap / sum(gt_counter.values())
    return (2 * precision * recall) / (precision + recall)


def normalize_tokens(value: Any) -> list[str]:
    if value is None:
        text = "null"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    text = text.lower()
    text = PUNCT_RE.sub(" ", text)
    text = ARTICLES_RE.sub(" ", text)
    text = " ".join(text.split())
    return text.split() if text else []


def compute_coverage_gate(modality: str, structure_coverage: float) -> float:
    if modality == "text":
        return 1.0 if structure_coverage >= 0.95 else 0.0
    return min(1.0, (structure_coverage / 0.90) ** 2) if structure_coverage > 0 else 0.0


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def f1_from_counts(overlap: int, pred_total: int, gt_total: int) -> float:
    if pred_total == 0 or gt_total == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / gt_total
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def build_summary(
    response_file: Path,
    records: list[RecordMetrics],
    data_quality: dict[str, int],
    bootstrap_samples: int,
    seed: int,
    top_k_errors: int,
    modality: str,
) -> dict[str, Any]:
    rows = [record.row for record in records]
    metrics = [
        "json_parse_success",
        "json_root_structured",
        "schema_valid_input",
        "schema_compliance",
        "leaf_value_em",
        "value_token_f1",
        "hier_path_recall",
        "path_set_f1",
        "type_precision",
        "strict_json_em",
    ]
    overall = summarize_rows(rows, metrics, bootstrap_samples, seed, weighted=False)
    overall_weighted = summarize_rows(
        rows, metrics, bootstrap_samples, seed, weighted=True
    )

    payload: dict[str, Any] = {
        "response_file": response_file.as_posix(),
        "num_records": len(rows),
        "model_ids": sorted({row["model_id"] for row in rows}),
        "data_quality": data_quality,
        "summary": {
            "overall": overall,
            "overall_weighted": overall_weighted,
        },
    }

    if modality == "image":
        error_analysis = summarize_error_paths(records, top_k_errors)
        if error_analysis:
            payload["error_analysis"] = error_analysis

    return payload


def summarize_rows(
    rows: list[dict[str, Any]],
    metrics: list[str],
    bootstrap_samples: int,
    seed: int,
    weighted: bool,
) -> dict[str, Any]:
    rng = random.Random(seed + (1 if weighted else 0))
    summary_metrics: dict[str, Any] = {}
    for metric in metrics:
        mean = (
            weighted_mean(rows, metric) if weighted else arithmetic_mean(rows, metric)
        )
        low, high = bootstrap_ci(rows, metric, bootstrap_samples, rng, weighted)
        summary_metrics[metric] = {
            "mean": mean,
            "ci95_low": low,
            "ci95_high": high,
            "metric_name": METRIC_DISPLAY_NAMES[metric],
        }

    category_scores: dict[str, Any] = {}
    for category_name, components in CATEGORIES.items():
        mean = category_mean(rows, components, weighted)
        low, high = bootstrap_category_ci(
            rows, components, bootstrap_samples, rng, weighted
        )
        category_scores[category_name] = {
            "mean": mean,
            "ci95_low": low,
            "ci95_high": high,
            "category_name": category_name,
            "components": components,
        }

    payload: dict[str, Any] = {
        "n": len(rows),
        "metrics": summary_metrics,
        "category_scores": category_scores,
    }
    if weighted:
        payload["weighting"] = "schema_complexity"
        payload["weight_field_priority"] = list(WEIGHT_FIELD_PRIORITY)
        payload["difficulty_weights"] = DIFFICULTY_WEIGHTS
    return payload


def arithmetic_mean(rows: list[dict[str, Any]], metric: str) -> float:
    if not rows:
        return 0.0
    return sum(row[metric] for row in rows) / len(rows)


def weighted_mean(rows: list[dict[str, Any]], metric: str) -> float:
    total_weight = sum(row["difficulty_weight"] for row in rows)
    if total_weight == 0:
        return 0.0
    return sum(row[metric] * row["difficulty_weight"] for row in rows) / total_weight


def category_mean(
    rows: list[dict[str, Any]], components: list[str], weighted: bool
) -> float:
    if not rows:
        return 0.0
    per_row = [
        {**row, "_category": sum(row[m] for m in components) / len(components)}
        for row in rows
    ]
    if weighted:
        return weighted_mean(per_row, "_category")
    return arithmetic_mean(per_row, "_category")


def bootstrap_ci(
    rows: list[dict[str, Any]],
    metric: str,
    samples: int,
    rng: random.Random,
    weighted: bool,
) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    estimates = []
    for _ in range(samples):
        sample_rows = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        estimates.append(
            weighted_mean(sample_rows, metric)
            if weighted
            else arithmetic_mean(sample_rows, metric)
        )
    estimates.sort()
    return percentile_bounds(estimates)


def bootstrap_category_ci(
    rows: list[dict[str, Any]],
    components: list[str],
    samples: int,
    rng: random.Random,
    weighted: bool,
) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    estimates = []
    for _ in range(samples):
        sample_rows = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        estimates.append(category_mean(sample_rows, components, weighted))
    estimates.sort()
    return percentile_bounds(estimates)


def percentile_bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    low_idx = max(0, math.floor(0.025 * (len(values) - 1)))
    high_idx = min(len(values) - 1, math.ceil(0.975 * (len(values) - 1)))
    return values[low_idx], values[high_idx]


def summarize_error_paths(
    records: list[RecordMetrics], top_k_errors: int
) -> dict[str, Any]:
    gt_counter = Counter()
    required_counter = Counter()
    for record in records:
        gt_counter.update(record.missing_gt_paths)
        required_counter.update(record.missing_required_paths)

    payload: dict[str, Any] = {}
    if gt_counter:
        payload["top_missing_gt_paths"] = [
            {"path": path, "count": count}
            for path, count in gt_counter.most_common(top_k_errors)
        ]
    if required_counter:
        payload["top_missing_required_paths"] = [
            {"path": path, "count": count}
            for path, count in required_counter.most_common(top_k_errors)
        ]
    return payload


if __name__ == "__main__":
    main()
