"""Unified inference entry point.

Usage
-----
    python -m sob.run --provider openrouter --modality text \
        --model-id google/gemma-4-31b-it --sample-size 5

Each provider module exposes a `run(records, config)` callable that returns
a list of (record, candidate, input_tokens, output_tokens, avg_time) tuples.
This driver dispatches on `config.provider`, then serializes each tuple
through `build_eval_record` and writes to the right response directory via
`resolve_output_path`.
"""

import argparse
from importlib import import_module

from sob.common.serialization import (
    build_eval_record,
    resolve_output_path,
    write_jsonl,
)
from sob.data_loader import load_data
from utils.config import InferenceConfig
from utils.logger import logger

PROVIDER_REGISTRY = {
    "openrouter": "sob.providers.openrouter",
    "vllm": "sob.providers.vllm",
    "openai": "sob.providers.openai_native",
    "anthropic": "sob.providers.anthropic_native",
    "gemini": "sob.providers.gemini_native",
}


def _provider_runner(name: str):
    if name not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider {name!r}")
    return import_module(PROVIDER_REGISTRY[name]).run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run model inference on the SOB benchmark.")
    p.add_argument(
        "--provider", default=None, help="openrouter|vllm|openai|anthropic|gemini"
    )
    p.add_argument("--modality", default=None, help="text|image|audio")
    p.add_argument("--model-id", default=None)
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--use-structured-decoding", action="store_true")
    return p.parse_args()


def _override_config(
    config: InferenceConfig, args: argparse.Namespace
) -> InferenceConfig:
    if args.provider:
        config.provider = args.provider
    if args.modality:
        config.modality = args.modality
    if args.model_id:
        config.model_id = args.model_id
    if args.sample_size is not None:
        config.sample_size = args.sample_size
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
    if args.use_structured_decoding:
        config.use_structured_decoding = True
    return config


def main() -> None:
    args = parse_args()
    config = _override_config(InferenceConfig(), args)

    logger.info(
        f"Run: provider={config.provider} modality={config.modality} "
        f"model_id={config.model_id} sample_size={config.sample_size}"
    )

    dataset = load_data(config)
    records = list(dataset)
    if config.sample_size is not None:
        records = records[: config.sample_size]
        logger.info(f"Sliced to first {config.sample_size} records.")

    runner = _provider_runner(config.provider)
    tuples = runner(records, config)

    eval_records = [
        build_eval_record(
            record=r,
            candidate=c,
            model_id=config.model_id,
            modality=config.modality,
            input_tokens=i,
            output_tokens=o,
            avg_time=t,
        )
        for (r, c, i, o, t) in tuples
    ]

    output_path = resolve_output_path(config.model_id, config.modality)
    write_jsonl(output_path, eval_records)
    logger.info(f"Saved {len(eval_records)} records to {output_path.resolve()}")


if __name__ == "__main__":
    main()
