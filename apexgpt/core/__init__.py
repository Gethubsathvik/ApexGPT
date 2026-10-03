"""Core cross-cutting concerns: config, paths, device, environment and seeding."""
from .config import Config, DataConfig, ModelConfig, TrainConfig
from .device import (HardwareProfile, autocast_dtype, available_backends,
                     configure_threads, cuda_available, describe_device,
                     dml_available, display_adapters, get_device, mps_available,
                     profile, rocm_build, total_ram_gb, total_vram_gb,
                     use_amp_by_default, use_checkpointing_by_default,
                     xpu_available)
from .environment import (AcceleratorInfo, EnvironmentReport, Requirement,
                          RuntimeSettings, apply_settings, bootstrap,
                          collect_requirements, environment_vars, resolve_settings,
                          scan, shell_exports, write_report)
from .paths import (CHECKPOINTS_DIR, DATA_DIR, MODELS_DIR, ROOT, RUNS_DIR,
                    TOKENIZER_DIR, describe)
from .seeding import set_seed

__all__ = [
    "Config", "DataConfig", "ModelConfig", "TrainConfig",
    "get_device", "cuda_available", "mps_available", "xpu_available",
    "dml_available", "rocm_build", "available_backends", "describe_device",
    "configure_threads", "autocast_dtype", "use_amp_by_default",
    "use_checkpointing_by_default", "total_vram_gb", "total_ram_gb", "profile",
    "HardwareProfile", "display_adapters",
    "AcceleratorInfo", "EnvironmentReport", "Requirement", "RuntimeSettings",
    "scan", "bootstrap", "resolve_settings", "apply_settings",
    "collect_requirements", "environment_vars", "shell_exports", "write_report",
    "set_seed", "ROOT", "DATA_DIR", "MODELS_DIR", "RUNS_DIR",
    "CHECKPOINTS_DIR", "TOKENIZER_DIR", "describe",
]
