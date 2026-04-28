from dataclasses import dataclass


@dataclass
class InferenceConfig:
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    model_id: str = (
        "openai/gpt-oss-20b"  # Change to the model you want to run inference with
    )
    modality: str = "text"  # "text" | "image" | "audio"
    provider: str = (
        "openrouter"  # "openrouter" | "vllm" | "openai" | "anthropic" | "gemini"
    )
    sample_size: int | None = None  # Set to None to run on the full dataset
    use_structured_decoding: bool = False
    disable_thinking: bool = True
    tensor_parallel_size: int = 1  # Adjust based on your GPU setup -> 1 for CPU or single GPU, >1 for multiple GPUs
    max_model_len: int = 8192
    max_tokens: int = 2048
    temperature: float = 0.0
    max_workers: int = 20
    max_retries: int = 5
    # Provider-specific knobs
    openai_reasoning_effort: str | None = (
        None  # reserved for future Responses API support
    )
    anthropic_checkpoint_every: int = 50
    gemini_max_concurrency: int = 20
    openrouter_extra_body: dict | None = (
        None  # e.g. {"provider": {"only": [...]}, "chat_template_kwargs": {...}}
    )
