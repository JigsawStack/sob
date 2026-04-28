from datasets import Dataset
from transformers import AutoTokenizer

from utils.logger import logger

DIFFICULTY_WEIGHTS = {
    "hard": 3,
    "medium": 2,
    "easy": 1,
}


def add_token_count(dataset: Dataset, tokenizer_name: str) -> Dataset:
    """Add `input_context_length` column using the given HF tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    logger.info(f"Tokenizer '{tokenizer_name}' loaded for token counting.")

    def _count(batch):
        tokens = tokenizer(
            batch["context"],
            truncation=False,
            add_special_tokens=False,
        )
        return {"input_context_length": [len(ids) for ids in tokens["input_ids"]]}

    dataset = dataset.map(_count, batched=True, batch_size=64)
    logger.info("Token counts added to dataset.")
    return dataset


def add_difficulty_weight(dataset: Dataset) -> Dataset:
    """Add `difficulty_weight` using `question_difficulty` if present,
    else falling back to `schema_complexity` (image/audio records)."""

    def _weight(example):
        difficulty = (
            example.get("question_difficulty")
            or example.get("schema_complexity")
            or "easy"
        )
        return {"difficulty_weight": DIFFICULTY_WEIGHTS.get(difficulty, 1)}

    dataset = dataset.map(_weight)
    logger.info("Difficulty weights added to dataset.")
    return dataset
