"""Build the unified leaderboard (paper Table 1) from per-source eval_summary.json files.

Discovery
---------
Model directories are auto-discovered by scanning
``data/evaluation/{text,image,audio}/<model_dir>/eval_summary.json``. Any
directory that contains an ``eval_summary.json`` is included; a model can be
present in any subset of the three modalities.

Display names are read from ``data/evaluation/display_names.json`` (a flat
``{dir_name: pretty_name}`` map). Directories without an entry fall back to
the directory name itself, so adding a new model is "drop the eval dir +
add one line to ``display_names.json``."

Aggregation
-----------
Uses the formula stated in the paper:

    bar_m_k = sum_u W_u * m^(w)_{k,u} / sum_u W_u

where m^(w)_{k,u} is the schema-complexity-weighted within-source mean
(read from ``summary.overall_weighted.metrics``) and W_u is the total
schema-complexity weight for source u: (W_t, W_i, W_a) = (13054, 602, 343).

Perfect Response Rate is aggregated over text + image only (audio omitted).

Overall (Adj.) = Overall (Raw) * coverage, where coverage = n_eval / 5324.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "data" / "evaluation"
DISPLAY_NAMES_FILE = EVAL_DIR / "display_names.json"

W_T, W_I, W_A = 13054, 602, 343
N_T, N_I, N_A = 5000, 209, 115
TOTAL_N = N_T + N_I + N_A  # 5324

MODALITIES = ("text", "image", "audio")

METRIC_KEYS = [
    ("leaf_value_em", "Val.Acc."),
    ("value_token_f1", "Faithful."),
    ("schema_compliance", "JSON Pass"),
    ("hier_path_recall", "Path Rec."),
    ("path_set_f1", "Str.Cov."),
    ("type_precision", "Type Saf."),
]


def load_display_names() -> dict[str, str]:
    """Load the dir_name -> display_name map; tolerate `_comment` entries."""
    if not DISPLAY_NAMES_FILE.exists():
        return {}
    raw = json.loads(DISPLAY_NAMES_FILE.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def discover_model_dirs() -> list[str]:
    """Return the sorted union of model directory names across all modalities.

    A directory counts if any modality has ``<dir>/eval_summary.json``. Hidden
    dirs and any leftover ``_audio_true`` ablation dirs are ignored as a guard.
    """
    seen: set[str] = set()
    for modality in MODALITIES:
        modality_dir = EVAL_DIR / modality
        if not modality_dir.is_dir():
            continue
        for child in modality_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith(".") or child.name.endswith("_audio_true"):
                continue
            if (child / "eval_summary.json").exists():
                seen.add(child.name)
    return sorted(seen)


def load_weighted_metrics(modality: str, model_dir: str):
    """Return dict[metric_key -> mean] from overall_weighted, or None if missing."""
    p = EVAL_DIR / modality / model_dir / "eval_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        s = json.load(f)
    metrics = s["summary"]["overall_weighted"]["metrics"]
    return {k: metrics[k]["mean"] for k in metrics}


def aggregate(values_with_weights):
    total_w = sum(w for _, w in values_with_weights)
    total = sum(v * w for v, w in values_with_weights)
    return total / total_w if total_w > 0 else 0.0


def compute_row(model_dir: str):
    text = load_weighted_metrics("text", model_dir)
    image = load_weighted_metrics("image", model_dir)
    audio = load_weighted_metrics("audio", model_dir)

    n_eval = (N_T if text else 0) + (N_I if image else 0) + (N_A if audio else 0)
    coverage = n_eval / TOTAL_N

    aggregates = {}
    for k, _ in METRIC_KEYS:
        vals = []
        if text is not None and k in text:
            vals.append((text[k], W_T))
        if image is not None and k in image:
            vals.append((image[k], W_I))
        if audio is not None and k in audio:
            vals.append((audio[k], W_A))
        if vals:
            aggregates[k] = aggregate(vals)

    perf_vals = []
    if text is not None and "strict_json_em" in text:
        perf_vals.append((text["strict_json_em"], W_T))
    if image is not None and "strict_json_em" in image:
        perf_vals.append((image["strict_json_em"], W_I))
    if perf_vals:
        aggregates["perfect"] = aggregate(perf_vals)

    keys = [k for k, _ in METRIC_KEYS] + ["perfect"]
    avail = [aggregates[k] for k in keys if k in aggregates]
    if not avail:
        return None
    raw = sum(avail) / len(avail)
    overall_adj = raw * coverage

    return {
        "overall_adj": overall_adj,
        "overall_raw": raw,
        "coverage": coverage,
        "modalities": [m for m, x in zip(MODALITIES, [text, image, audio]) if x],
        "value_accuracy": aggregates.get("leaf_value_em"),
        "faithfulness": aggregates.get("value_token_f1"),
        "json_pass": aggregates.get("schema_compliance"),
        "path_recall": aggregates.get("hier_path_recall"),
        "structure_cov": aggregates.get("path_set_f1"),
        "type_safety": aggregates.get("type_precision"),
        "perfect": aggregates.get("perfect"),
    }


JSON_KEY_MAP = {
    "overall_adj": "overall",
    "json_pass": "json_pass_rate",
    "structure_cov": "structure_coverage",
    "perfect": "perfect_response",
}
JSON_ROW_KEYS = [
    "model",
    "overall",
    "value_accuracy",
    "faithfulness",
    "json_pass_rate",
    "path_recall",
    "structure_coverage",
    "type_safety",
    "perfect_response",
    "modalities",
]


def _to_json_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        out[JSON_KEY_MAP.get(k, k)] = v
    return {k: out.get(k) for k in JSON_ROW_KEYS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="leaderboard.json",
        help="Path to write the leaderboard JSON (default: leaderboard.json)",
    )
    parser.add_argument(
        "--print-latex",
        action="store_true",
        help="Also print LaTeX rows for paper Table 1.",
    )
    args = parser.parse_args()

    display_names = load_display_names()
    model_dirs = discover_model_dirs()
    if not model_dirs:
        print(
            "error: no model directories with eval_summary.json found under data/evaluation/",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = []
    skipped: list[str] = []
    for model_dir in model_dirs:
        row = compute_row(model_dir)
        if row is None:
            skipped.append(model_dir)
            continue
        row["model"] = display_names.get(model_dir, model_dir)
        row["model_dir"] = model_dir
        rows.append(row)

    rows.sort(key=lambda r: -r["overall_adj"])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema_version": 1,
        "rows": [_to_json_row(r) for r in rows],
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(rows)} rows -> {out_path}")
    if skipped:
        print(f"  (skipped {len(skipped)} dirs with no metrics: {', '.join(skipped)})")
    print()

    cols = [
        "model",
        "overall_adj",
        "value_accuracy",
        "faithfulness",
        "json_pass",
        "path_recall",
        "structure_cov",
        "type_safety",
        "perfect",
    ]
    widths = [25, 12, 10, 10, 10, 10, 10, 10, 10]
    print("".join(f"{c:<{w}}" for c, w in zip(cols, widths)))
    for r in rows:
        cells = []
        for c, w in zip(cols, widths):
            v = r.get(c)
            if isinstance(v, str):
                cells.append(f"{v:<{w}}")
            elif v is None:
                cells.append(f"{'-':<{w}}")
            else:
                cells.append(f"{v:.4f}".ljust(w))
        print("".join(cells))

    if args.print_latex:
        print("\n% --- LaTeX rows for Table 1 (overall_leaderboard) ---")
        for r in rows:
            perf = f"{r['perfect']:.3f}" if r["perfect"] is not None else "--"
            print(
                f"{r['model']:<22} & "
                f"{r['overall_adj']:.3f} & {r['value_accuracy']:.3f} & "
                f"{r['faithfulness']:.3f} & {r['json_pass']:.3f} & "
                f"{r['path_recall']:.3f} & {r['structure_cov']:.3f} & "
                f"{r['type_safety']:.3f} & {perf} \\\\"
            )


if __name__ == "__main__":
    main()
