"""Resolve which trained checkpoint to deploy.

LeRobot writes outputs/train/<job>/checkpoints/<zero-padded step>/pretrained_model
and keeps a `last` symlink pointing at the newest.
"""
from pathlib import Path

CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIR = "pretrained_model"


def resolve_checkpoint(output_dir: str | Path, step: str | None = None) -> Path:
    """Path of the checkpoint to load: `step` if given, else `last`, else the highest step."""
    checkpoints = Path(output_dir) / CHECKPOINTS_DIR
    if not checkpoints.is_dir():
        raise FileNotFoundError(f"No checkpoints directory under '{output_dir}'. Has training run?")

    available = sorted(
        (path for path in checkpoints.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if step is not None:
        chosen = checkpoints / step
        if not (chosen / PRETRAINED_MODEL_DIR).is_dir():
            names = ", ".join(path.name for path in available) or "none"
            raise FileNotFoundError(f"No checkpoint '{step}' under '{checkpoints}'. Available: {names}")
        return chosen / PRETRAINED_MODEL_DIR

    last = checkpoints / LAST_CHECKPOINT_LINK
    if (last / PRETRAINED_MODEL_DIR).is_dir():
        return last.resolve() / PRETRAINED_MODEL_DIR
    if not available:
        raise FileNotFoundError(f"No checkpoints under '{checkpoints}'. Has training run?")
    return available[-1] / PRETRAINED_MODEL_DIR
