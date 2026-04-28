import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from sob.common.prompts import SYSTEM_PROMPT, build_user_message
from sob.common.schema_utils import parse_if_string, normalize_schema_strict
from utils.config import InferenceConfig
from utils.logger import logger


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
                logger.error(f"Failed: {str(record.get('record_id', ''))[:12]}... ({e})")

    # Record the effective schema used for evaluation: strict schema only when
    # it was sent to OpenAI's schema-constrained mode.
    effective_schema = strict_schema if config.use_structured_decoding else schema
    
    return record, candidate, effective_schema, input_tokens, output_tokens


def run(records: list[dict], config: InferenceConfig) -> list[tuple[dict, object, int, int, float]]:
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
            ex.submit(_infer_one, client, r, config): i
            for i, r in enumerate(records)
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
