import json
import time

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from sob.common.prompts import SYSTEM_PROMPT
from sob.common.schema_utils import parse_if_string, extract_json
from utils.config import InferenceConfig
from utils.logger import logger


def run(
    records: list[dict], config: InferenceConfig
) -> list[tuple[dict, object, int, int, float]]:
    """Run local vLLM inference over `records`."""
    llm = LLM(
        model=config.model_id,
        tensor_parallel_size=config.tensor_parallel_size,
        max_model_len=config.max_model_len,
        trust_remote_code=True,
    )
    logger.info(
        f"vLLM provider ready: model={config.model_id} "
        f"tp={config.tensor_parallel_size} structured={config.use_structured_decoding}"
    )

    tokenizer = llm.get_tokenizer()
    prompts: list[str] = []
    params_list: list = []
    schemas: list[dict] = []

    for record in tqdm(records, desc="Building prompts"):
        schema = parse_if_string(record["json_schema"])
        schemas.append(schema)
        schema_str = json.dumps(schema, indent=2)
        user_msg = (
            f"Context:\n{record['context']}\n\n"
            f"Question: {record['question']}\n\n"
            f"Respond with JSON matching this schema:\n{schema_str}\n\n"
            f"Return ONLY the JSON object.\n\n"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if config.disable_thinking:
            template_kwargs["enable_thinking"] = False
        prompts.append(tokenizer.apply_chat_template(messages, **template_kwargs))

        sp_kwargs = {"max_tokens": config.max_tokens, "temperature": config.temperature}
        if config.use_structured_decoding:
            sp_kwargs["structured_outputs"] = StructuredOutputsParams(json=schema)
        params_list.append(SamplingParams(**sp_kwargs))

    logger.info(f"Built {len(prompts)} prompts. Running batch inference...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params=params_list)
    total_time = time.time() - start
    avg_time = round(total_time / max(1, len(records)), 4)

    results: list[tuple[dict, object, int, int, float]] = []
    n_valid = n_invalid = 0
    total_input = total_output = 0

    for record, output, schema in zip(records, outputs, schemas):
        raw = output.outputs[0].text
        if config.use_structured_decoding:
            try:
                candidate = json.loads(raw)
            except json.JSONDecodeError:
                candidate = raw
        else:
            candidate = extract_json(raw)

        input_toks = len(output.prompt_token_ids)
        output_toks = len(output.outputs[0].token_ids)
        total_input += input_toks
        total_output += output_toks

        if isinstance(candidate, dict):
            n_valid += 1
        else:
            n_invalid += 1

        results.append((record, candidate, input_toks, output_toks, avg_time))

    logger.info(
        f"vLLM done. {len(records)} records in {total_time:.1f}s ({avg_time}s/record). "
        f"valid={n_valid} invalid={n_invalid} input_tokens={total_input:,} output_tokens={total_output:,}"
    )
    return results
