import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
from tqdm import tqdm

from sob.common.checkpoint import JsonlCheckpoint, checkpoint_path_for
from sob.common.prompts import SYSTEM_PROMPT, build_user_message
from sob.common.schema_utils import extract_json, parse_if_string
from sob.common.serialization import build_eval_record
from utils.config import InferenceConfig
from utils.logger import logger


def _backoff_seconds(error: Exception, attempt: int) -> int:
    """Rate-limit-aware exponential backoff.

    - RateLimitError: more aggressive (2^(n+2) = 4, 8, 16, 32, ...)
    - Other errors: gentler (2^(n+1) = 2, 4, 8, ...)
    """
    if isinstance(error, anthropic.RateLimitError):
        return 2 ** (attempt + 2)
    return 2 ** (attempt + 1)


def _infer_one(
    client: anthropic.Anthropic,
    record: dict,
    config: InferenceConfig,
) -> tuple[dict, object | None, int, int]:
    schema = parse_if_string(record.get("json_schema"))
    user_msg = build_user_message(record, schema=schema)

    candidate = None
    input_tokens = output_tokens = 0

    for attempt in range(config.max_retries):
        try:
            response = client.messages.create(
                model=config.model_id,
                max_tokens=config.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                temperature=config.temperature,
            )
            # Anthropic returns a list of content blocks; text lives on the
            # first one (TextBlock) for plain JSON tasks.
            raw = ""
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    raw = block.text
                    break
            if response.usage:
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

            candidate = extract_json(raw)
            break
        except Exception as e:
            if attempt < config.max_retries - 1:
                wait = _backoff_seconds(e, attempt)
                logger.warning(
                    f"Retry {attempt + 1}/{config.max_retries} for "
                    f"{str(record.get('record_id', ''))[:12]}... waiting {wait}s "
                    f"({type(e).__name__})"
                )
                time.sleep(wait)
            else:
                logger.error(f"Failed: {str(record.get('record_id', ''))[:12]}... ({e})")

    return record, candidate, input_tokens, output_tokens


def run(records: list[dict], config: InferenceConfig) -> list[tuple[dict, object, int, int, float]]:
    """Run Anthropic native inference with a JsonlCheckpoint.

    On resume, records already in the checkpoint are skipped. The checkpoint
    file is timestamped; if you want to resume a specific prior run, pass its
    path via `config.anthropic_checkpoint_path` (not yet implemented — a
    feature request flag).
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    ckpt_path = checkpoint_path_for(config.model_id)
    logger.info(
        f"Anthropic provider ready: model={config.model_id} "
        f"checkpoint={ckpt_path} every={config.anthropic_checkpoint_every}"
    )

    ckpt = JsonlCheckpoint(ckpt_path, every=config.anthropic_checkpoint_every)
    done = ckpt.load()
    if done:
        logger.info(f"Loaded {len(done)} completed records from checkpoint.")

    remaining_indexed = [
        (i, r) for i, r in enumerate(records)
        if str(r.get("record_id")) not in done
    ]
    logger.info(
        f"{len(records)} total records; {len(remaining_indexed)} to run, "
        f"{len(records) - len(remaining_indexed)} already done."
    )

    results: list = [None] * len(records)

    # Fill in already-completed entries (re-project back to the tuple format).
    for i, r in enumerate(records):
        prev = done.get(str(r.get("record_id")))
        if prev is None:
            continue
        out_meta = prev.get("output", {})
        ei = prev.get("eval_info", {})
        results[i] = (
            r,
            out_meta.get("candidate_response"),
            ei.get("input_tokens", 0),
            ei.get("output_tokens", 0),
        )

    n_valid = sum(1 for x in results if x is not None and isinstance(x[1], dict))
    n_failed = 0
    n_invalid = 0

    start = time.time()
    with ckpt:
        with ThreadPoolExecutor(max_workers=config.max_workers) as ex:
            futures = {
                ex.submit(_infer_one, client, r, config): i
                for i, r in remaining_indexed
            }
            pbar = tqdm(as_completed(futures), total=len(remaining_indexed), desc="Anthropic")
            for fut in pbar:
                idx = futures[fut]
                record, candidate, in_tok, out_tok = fut.result()
                results[idx] = (record, candidate, in_tok, out_tok)

                if isinstance(candidate, dict):
                    n_valid += 1
                elif candidate is None:
                    n_failed += 1
                else:
                    n_invalid += 1

                # Checkpoint the intermediate record. We use a zero avg_time
                # here because per-record wall-clock isn't meaningful under
                # ThreadPool contention; the caller patches in the true
                # `avg_time` after the whole run finishes.
                ckpt_record = build_eval_record(
                    record=record,
                    candidate=candidate,
                    model_id=config.model_id,
                    modality=config.modality,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    avg_time=0.0,
                )
                ckpt.append(ckpt_record)
                pbar.set_postfix(valid=n_valid, bad_json=n_invalid, failed=n_failed)

    total_time = time.time() - start
    avg_time = round(total_time / max(1, len(remaining_indexed)), 4) if remaining_indexed else 0.0
    logger.info(
        f"Anthropic done. {len(records)} records total "
        f"(newly inferred: {len(remaining_indexed)} in {total_time:.1f}s, "
        f"{avg_time}s/record). valid={n_valid} invalid={n_invalid} failed={n_failed}"
    )

    return [(r, c, i, o, avg_time) for (r, c, i, o) in results if r is not None]
