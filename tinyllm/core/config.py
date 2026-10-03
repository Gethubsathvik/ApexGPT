"""Typed configuration for every layer of TinyLLM."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .paths import DATA_DIR


def _default_dataset() -> str:
    """Corpus key from the environment, so scripts and notebooks can switch."""
    return os.environ.get("TINYLLM_DATASET", "wikipedia")


@dataclass
class DataConfig:
    vocab_size: int = 50257            # GPT-2 BPE vocabulary
    block_size: int = 256              # training context window
    batch_size: int = 8
    val_batch_size: int = 8            # honored by the data service
    train_split: float = 0.8           # honored by the data service
    dataset: str = field(default_factory=_default_dataset)
    # Each corpus gets its own directory. Pass an explicit path to override.
    raw_dir: Path | None = None
    binary_dir: Path | None = None
    dataset_name: str = "tinyllm-corpus"

    def __post_init__(self) -> None:
        # Not a dataclass field: asdict() must not serialise it.
        self._dirs_are_default = (self.raw_dir is None and self.binary_dir is None)
        if self.raw_dir is None:
            self.raw_dir = DATA_DIR / "raw" / self.dataset
        if self.binary_dir is None:
            self.binary_dir = DATA_DIR / "binary" / self.dataset

    def select_dataset(self, name: str) -> None:
        """Switch corpus, moving the data directories with it.

        Assigning ``.dataset`` directly would leave the directories pointing at
        the previous corpus, which silently reads the wrong tokens.
        """
        self.dataset = name
        if self._dirs_are_default:
            self.raw_dir = DATA_DIR / "raw" / self.dataset
            self.binary_dir = DATA_DIR / "binary" / self.dataset

    @property
    def corpus_path(self) -> Path:
        """The plain-text corpus this dataset is built from."""
        return self.raw_dir / f"{self.dataset}.txt"

    @property
    def train_path(self) -> Path:
        return self.binary_dir / "train.bin"

    @property
    def val_path(self) -> Path:
        return self.binary_dir / "val.bin"

    def is_ready(self) -> bool:
        return (self.train_path.exists() and self.val_path.exists()
                and self.train_path.stat().st_size > 0
                and self.val_path.stat().st_size > 0)


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True

    def validate(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")


@dataclass
class TrainConfig:
    max_iters: int = 4000
    epochs: int = 3                    # used to derive max_iters when set
    eval_interval: int = 200
    eval_iters: int = 40
    checkpoint_interval: int = 500
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 1337
    use_amp: bool = True
    use_checkpointing: bool = True
    num_threads: int | None = None      # None -> all available cores
    run_name: str = "gpt-tinyllm"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)
