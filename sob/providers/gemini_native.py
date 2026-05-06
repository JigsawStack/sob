"""Gemini provider.

Reasoning / thinking notes
--------------------------
The default `thinking_budget=0` below fully disables thinking on **Gemini 2.5
Flash** (the model used in the published leaderboard). It will NOT work for
every other Gemini variant — each enforces a different floor and a different
parameter:

  - 2.5 Flash       → thinking_budget = 0    (full disable supported)
  - 2.5 Flash-Lite  → thinking_budget ≥ 512  (no full disable)
  - 2.5 Pro         → thinking_budget ≥ 128  (no full disable)
  - 3 Flash         → thinking_level = "minimal"   (different param entirely)
  - 3 Pro / 3.1 Pro → thinking_level = "low"       ("minimal" not exposed)

If you run a model from any of the other rows above, edit the
`ThinkingConfig` call below to match — passing `thinking_budget=0` to a Pro or
Flash-Lite model will error, and 2.5-style budgets are silently ignored on
Gemini 3.
"""

import asyncio
import os
import time

from google import genai
from google.genai import types
from tqdm.auto import tqdm

from sob.common.prompts import SYSTEM_PROMPT, build_user_message
from sob.common.schema_utils import (
    extract_json,
    parse_if_string,
    sanitize_schema_for_gemini,
)
from utils.config import InferenceConfig
from utils.logger import logger


def _thinking_config_for(model_id: str) -> "types.ThinkingConfig":
    """Pick the right ThinkingConfig knobs for the model's parameter shape.

    Gemini 3 introduced `thinking_level` and dropped `thinking_budget` for
    Pro/Flash; 2.5 Pro and 2.5 Flash-Lite enforce non-zero floors. Passing
    the wrong knob to any of these generations errors out. See the floors
    documented in the module docstring.
    """
    mid = model_id.lower().removeprefix("models/")
    if mid.startswith("gemini-3"):
        level = "minimal" if "flash" in mid else "low"
        return types.ThinkingConfig(thinking_level=level)
    if "2.5-pro" in mid:
        return types.ThinkingConfig(thinking_budget=128)
    if "2.5-flash-lite" in mid:
        return types.ThinkingConfig(thinking_budget=512)
    return types.ThinkingConfig(thinking_budget=0)


async def _infer_one(
    aclient,
    record: dict,
    config: InferenceConfig,
    sem: asyncio.Semaphore,
) -> tuple[dict, object | None, dict, int, int]:
    raw_schema = parse_if_string(record.get("json_schema"))
    sanitized = sanitize_schema_for_gemini(raw_schema)

    user_msg = build_user_message(record, schema=raw_schema)
    config_kwargs = {
        "system_instruction": SYSTEM_PROMPT,
        "temperature": config.temperature,
        "max_output_tokens": config.max_tokens,
        "response_mime_type": "application/json",
    }
    if config.disable_thinking:
        config_kwargs["thinking_config"] = _thinking_config_for(config.model_id)
    if config.use_structured_decoding:
        config_kwargs["response_schema"] = sanitized

    input_tokens = output_tokens = 0
    candidate = None
    async with sem:
        response = None
        raw = ""
        try:
            response = await aclient.models.generate_content(
                model=config.model_id,
                contents=user_msg,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            raw = response.text or ""
        except Exception as e:
            logger.error(f"Failed: {str(record.get('record_id', ''))[:12]}... ({e})")
            raw = ""

    if response is not None:
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0

    candidate = extract_json(raw) if raw else None
    effective_schema = sanitized if config.use_structured_decoding else raw_schema
    return record, candidate, effective_schema, input_tokens, output_tokens


async def _run_async(records: list[dict], config: InferenceConfig):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    aclient = client.aio
    sem = asyncio.Semaphore(config.gemini_max_concurrency)

    logger.info(
        f"Gemini native provider ready: model={config.model_id} "
        f"concurrency={config.gemini_max_concurrency} "
        f"structured={config.use_structured_decoding}"
    )

    tasks = [_infer_one(aclient, r, config, sem) for r in records]
    results_by_idx: list = [None] * len(records)

    start = time.time()
    pbar = tqdm(total=len(tasks), desc="Gemini")
    for fut in asyncio.as_completed(tasks):
        rec, candidate, sanitized, in_tok, out_tok = await fut
        # We don't know the original index from `as_completed`, so match by
        # record_id; cheaper than per-task wrapping.
        rid = rec.get("record_id")
        for i, r in enumerate(records):
            if r.get("record_id") == rid and results_by_idx[i] is None:
                results_by_idx[i] = (rec, candidate, sanitized, in_tok, out_tok)
                break
        pbar.update(1)
    pbar.close()

    total_time = time.time() - start
    avg_time = round(total_time / max(1, len(records)), 4)
    logger.info(
        f"Gemini done. {len(records)} records in {total_time:.1f}s "
        f"({avg_time}s/record)."
    )

    out = []
    for entry in results_by_idx:
        if entry is None:
            continue
        rec, candidate, sanitized, in_tok, out_tok = entry
        rec = dict(rec)
        rec["json_schema"] = sanitized
        out.append((rec, candidate, in_tok, out_tok, avg_time))
    return out


def run(
    records: list[dict], config: InferenceConfig
) -> list[tuple[dict, object, int, int, float]]:
    """Run Gemini inference (wraps async internals in a sync call).

    Note: if called from within an existing event loop (e.g. a Jupyter
    notebook without nest_asyncio), this will fail. Use `asyncio.run`
    directly from a notebook instead.
    """
    return asyncio.run(_run_async(records, config))
