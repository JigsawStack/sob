import json
import os
import time
from pathlib import Path
from typing import Any


def checkpoint_path_for(model_id: str) -> Path:
    """Modal-aware checkpoint path, timestamped to avoid cross-run collisions."""
    model_slug = model_id.replace("/", "_")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if os.getenv("MODAL_TASK_ID"):
        base = Path("/mnt/hf-cache/checkpoints")
    else:
        base = Path("data/checkpoints")
    return base / f"results_{model_slug}_{stamp}.jsonl"


class JsonlCheckpoint:
    """Append-only JSONL checkpoint with periodic flush.

    Used by the Anthropic provider for long runs where the full retry shape
    makes mid-run failures expensive. Other providers use line-count-based
    resumption instead.

    Records are keyed by `metadata.record_id` in `load()` so a resume call
    can skip already-completed records.
    """

    def __init__(self, path: Path, every: int = 50):
        self.path = path
        self.every = max(1, every)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._since_flush = 0

    def load(self) -> dict[str, dict]:
        done: dict[str, dict] = {}
        if not self.path.exists():
            return done
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rid = rec.get("metadata", {}).get("record_id")
                    if rid is not None:
                        done[str(rid)] = rec
                except json.JSONDecodeError:
                    pass  # skip corrupt lines from a prior aborted run
        return done

    def __enter__(self):
        self._file = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def append(self, result: dict[str, Any]) -> None:
        if self._file is None:
            raise RuntimeError("JsonlCheckpoint.append called outside `with` block")
        self._file.write(json.dumps(result) + "\n")
        self._since_flush += 1
        if self._since_flush >= self.every:
            self._file.flush()
            self._since_flush = 0
