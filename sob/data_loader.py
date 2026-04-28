import os

from datasets import Dataset, load_dataset

from sob.common.tokenize import add_difficulty_weight, add_token_count
from utils.config import InferenceConfig
from utils.logger import logger

# Maps each modality to its HF dataset config + split. Text records live in
# the default config's "test" split; image and audio live under dedicated
# configs with a single "train" split each.
MODALITY_DATASET = {
    "text": {"config_name": None, "split": "test"},
    "image": {"config_name": "image", "split": "train"},
    "audio": {"config_name": "audio", "split": "train"},
}


def load_data(config: InferenceConfig | None = None) -> Dataset:
    config = config or InferenceConfig()
    modality = config.modality
    if modality not in MODALITY_DATASET:
        raise ValueError(
            f"Unknown modality {modality!r}. Expected one of {list(MODALITY_DATASET)}"
        )

    spec = MODALITY_DATASET[modality]
    load_kwargs = {"split": spec["split"]}
    if spec["config_name"]:
        load_kwargs["name"] = spec["config_name"]
    hf_token = os.getenv("HF_TOKEN") or os.getenv("hf_token")
    if hf_token:
        load_kwargs["token"] = hf_token

    dataset = load_dataset("interfaze-ai/sob", **load_kwargs)
    logger.info(
        f"Dataset loaded for modality={modality!r} "
        f"(config={spec['config_name']}, split={spec['split']})"
    )
    logger.info(f"Dataset size: {len(dataset)}")

    dataset = add_token_count(dataset, config.sentence_transformer_model)
    dataset = add_difficulty_weight(dataset)
    return dataset
