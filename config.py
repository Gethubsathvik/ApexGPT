"""Central configuration for TinyLLM."""
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Bulk data (parquet shards + token binaries) is large. The project ships with a
# data/ folder as required, but the storage root can be moved to another drive
# with the TINYLLM_DATA_DIR environment variable when the default drive is full.
DATA_ROOT = Path(os.environ.get("TINYLLM_DATA_DIR", ROOT / "data")).resolve()

DATA_DIR = DATA_ROOT
MODELS_DIR = ROOT / "models"
SCRIPTS_DIR = ROOT / "scripts"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RUNS_DIR = MODELS_DIR / "runs"

for _d in (DATA_DIR, MODELS_DIR, SCRIPTS_DIR, CHECKPOINTS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    vocab_size: int = 50257          # GPT-2 BPE
    block_size: int = 256            # training context window
    batch_size: int = 8
    val_batch_size: int = 8
    train_split: float = 0.8
    raw_dir: Path = DATA_DIR / "raw"
    binary_dir: Path = DATA_DIR / "binary"
    dataset_name: str = "tinyllm-corpus"


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True


@dataclass
class TrainConfig:
    max_iters: int = 4000
    epochs: int = 3
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
    num_threads: int = 8
    run_name: str = "gpt-tinyllm"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)


def set_seed(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("PYTHONHASHSEED", str(seed))