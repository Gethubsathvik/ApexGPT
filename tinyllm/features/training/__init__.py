"""Training feature: the training loop, optimizer schedule and checkpoints."""
from .service import PRESETS, TrainResult, plot_curves, show_history, train

__all__ = ["train", "TrainResult", "PRESETS", "plot_curves", "show_history"]