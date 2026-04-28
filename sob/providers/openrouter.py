import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from sob.common.prompts import SYSTEM_PROMPT, build_user_message
from sob.common.schema_utils import parse_if_string
from utils.config import InferenceConfig
from utils.logger import logger


def _infer_one(
    client: OpenAI,
    record: dict,
    config: InferenceConfig,
) -> tuple[dict, object | None, int, int]:
    schema = parse_if_string(record.get("json_schema"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(record, schema=schema)},
    ]

    extra_body: dict = {}
    if config.openrouter_extra_body:
        extra_body.update(config.openrouter_extra_body)
    # OpenRouter routes the per-provider reasoning toggle for thinking-capable
    # models (gpt-5 reasoning_effort, qwen3 enable_thinking, etc.). User-provided
    # extra_body wins via the order above.
    if config.disable_thinking:
        extra_body.setdefault("reasoning", {"effort": "minimal", "exclude": True})

    candidate = None
    input_tokens = 0
    output_tokens = 0

    for attempt in range(config.max_retries):
        try:
            kwargs = dict(
                model=config.model_id,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                response_format={"type": "json_object"},
            )
            if extra_body:
                kwargs["extra_body"] = extra_body

            response = client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content
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

    return record, candidate, input_tokens, output_tokens


def run(records: list[dict], config: InferenceConfig) -> list[tuple[dict, object, int, int, float]]:
    """Run OpenRouter inference over `records`.

    Returns a list of tuples: (record, candidate_response, input_tokens,
    output_tokens, avg_time_per_record). The driver passes each tuple into
    `build_eval_record` for final serialization.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    logger.info(f"OpenRouter provider ready: model={config.model_id}")

    results: list = [None] * len(records)
    n_valid = n_failed = n_invalid_json = 0

    start = time.time()
    with ThreadPoolExecutor(max_workers=config.max_workers) as ex:
        futures = {
            ex.submit(_infer_one, client, r, config): i
            for i, r in enumerate(records)
        }
        pbar = tqdm(as_completed(futures), total=len(records), desc="OpenRouter")
        for fut in pbar:
            idx = futures[fut]
            record, candidate, in_tok, out_tok = fut.result()
            results[idx] = (record, candidate, in_tok, out_tok)
            if isinstance(candidate, dict):
                n_valid += 1
            elif candidate is None:
                n_failed += 1
            else:
                n_invalid_json += 1
            pbar.set_postfix(valid=n_valid, bad_json=n_invalid_json, failed=n_failed)

    total_time = time.time() - start
    avg_time = round(total_time / max(1, len(records)), 4)
    logger.info(
        f"OpenRouter done. {len(records)} records in {total_time:.1f}s "
        f"({avg_time}s/record). valid={n_valid} invalid={n_invalid_json} failed={n_failed}"
    )

    return [(r, c, i, o, avg_time) for (r, c, i, o) in results if r is not None]
