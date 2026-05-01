"""OpenAI provider.

Reasoning / thinking notes
--------------------------
The chat completions endpoint accepts different knobs for different families,
and passing the wrong one returns 400. Current behaviour matches the gates
below:

  - GPT-5 / GPT-5-Mini / GPT-5-Nano  → reasoning_effort = "minimal".
    Temperature is NOT accepted; we omit it.
  - o-series (o1, o3, o4-mini, ...)  → reasoning_effort = "minimal".
    Most also refuse temperature; current code does not pass it for them
    either (the `not startswith("gpt-5")` gate sends temperature, which o1/o3
    will reject — patch the gate if you add an o-series model).
  - GPT-4.1 / GPT-4o / GPT-4o-mini   → standard chat. Reasoning effort is not
    exposed and would error if passed; temperature is required.

If you add a new model from any family, double-check both gates before
running — silent acceptance of a 400 wastes the whole sweep.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from sob.common.prompts import SYSTEM_PROMPT, build_user_message
from sob.common.schema_utils import parse_if_string, normalize_schema_strict
from utils.config import InferenceConfig
from utils.logger import logger


_GPT5_DOTTED = re.compile(r"^gpt-5\.\d")


def _min_reasoning_effort(model_id: str) -> str | None:
    """Return the lowest reasoning_effort string the model accepts, or None."""
    if _GPT5_DOTTED.match(model_id):
        return "none"
    if model_id.startswith("gpt-5") or model_id.startswith("o"):
        return "minimal"
    return None


def _infer_one(
    client: OpenAI,
    record: dict,
    config: InferenceConfig,
) -> tuple[dict, object | None, dict, int, int]:
    schema = parse_if_string(record.get("json_schema"))
    strict_schema = normalize_schema_strict(schema)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(record, schema=schema)},
    ]

    request_kwargs: dict = dict(
        model=config.model_id,
        messages=messages,
        max_completion_tokens=config.max_tokens,
        response_format={"type": "json_object"},
    )
    if config.use_structured_decoding:
        request_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": strict_schema,
                "strict": True,
            },
        }

    # gpt-5 family doesn't accept temperature; other models do.
    if not config.model_id.startswith("gpt-5"):
        request_kwargs["temperature"] = config.temperature

    # Reasoning models: pick the lowest effort value the model accepts.
    # gpt-5 / gpt-5-mini / gpt-5-nano / o-series take "minimal"; gpt-5.1+
    # (gpt-5.5, gpt-5.2, ...) replaced "minimal" with "none". Wrong value
    # 400s, so detect by version.
    effort = _min_reasoning_effort(config.model_id)
    if config.disable_thinking and effort is not None:
        request_kwargs["reasoning_effort"] = effort

    candidate = None
    input_tokens = output_tokens = 0

    for attempt in range(config.max_retries):
        try:
            response = client.chat.completions.create(**request_kwargs)
            raw = response.choices[0].message.content or ""
            if response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
            try:
                candidate = json.loads(raw)
            except json.JSONDecodeError:
                candidate = raw
            break
        except Exception as e:
            if attempt < config.max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    f"Retry {attempt + 1}/{config.max_retries} for "
                    f"{str(record.get('record_id', ''))[:12]}... waiting {wait}s"
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"Failed: {str(record.get('record_id', ''))[:12]}... ({e})"
                )

    # Record the effective schema used for evaluation: strict schema only when
    # it was sent to OpenAI's schema-constrained mode.
    effective_schema = strict_schema if config.use_structured_decoding else schema

    return record, candidate, effective_schema, input_tokens, output_tokens


def run(
    records: list[dict], config: InferenceConfig
) -> list[tuple[dict, object, int, int, float]]:
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://api.openai.com/v1",
    )
    logger.info(f"OpenAI native provider ready: model={config.model_id}")

    results: list = [None] * len(records)
    n_valid = n_failed = n_invalid = 0
    # We keep the schema-per-record inside the tuple so serialization records
    # the effective schema for the selected output mode.
    records_with_schema: list = [None] * len(records)

    start = time.time()
    with ThreadPoolExecutor(max_workers=config.max_workers) as ex:
        futures = {
            ex.submit(_infer_one, client, r, config): i for i, r in enumerate(records)
        }
        pbar = tqdm(as_completed(futures), total=len(records), desc="OpenAI")
        for fut in pbar:
            idx = futures[fut]
            record, candidate, effective_schema, in_tok, out_tok = fut.result()
            results[idx] = (record, candidate, in_tok, out_tok)
            records_with_schema[idx] = (record, effective_schema)
            if isinstance(candidate, dict):
                n_valid += 1
            elif candidate is None:
                n_failed += 1
            else:
                n_invalid += 1
            pbar.set_postfix(valid=n_valid, bad_json=n_invalid, failed=n_failed)

    total_time = time.time() - start
    avg_time = round(total_time / max(1, len(records)), 4)
    logger.info(
        f"OpenAI done. {len(records)} records in {total_time:.1f}s "
        f"({avg_time}s/record). valid={n_valid} invalid={n_invalid} failed={n_failed}"
    )

    # Merge the effective schema into the record so downstream serialization
    # evaluates against the same contract used by this run.
    out = []
    for (r, c, i, o), (_, s) in zip(results, records_with_schema):
        if r is None:
            continue
        r = dict(r)
        r["json_schema"] = s
        out.append((r, c, i, o, avg_time))
    return out
