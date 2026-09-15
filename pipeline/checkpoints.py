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
        # Validate step is numeric
        try:
            step_int = int(step)
        except ValueError as error:
            names = ", ".join(path.name for path in available) or "none"
            raise ValueError(
                f"'{step}' is not a valid checkpoint step (must be numeric). Available: {names}"
            ) from error
        # Match by integer value, not string equality
        for path in available:
            if int(path.name) == step_int:
                return path / PRETRAINED_MODEL_DIR
        names = ", ".join(path.name for path in available) or "none"
        raise FileNotFoundError(f"No checkpoint for step {step_int} under '{checkpoints}'. Available: {names}")

    last = checkpoints / LAST_CHECKPOINT_LINK
    # Check if last symlink exists, is valid, and points to a checkpoint with pretrained_model
    if last.is_symlink():
        try:
            resolved = last.resolve()
            if (resolved / PRETRAINED_MODEL_DIR).is_dir():
                return resolved / PRETRAINED_MODEL_DIR
        except (OSError, RuntimeError):
            pass  # Dangling symlink or resolution error; fall through to available

    if not available:
        raise FileNotFoundError(f"No checkpoints under '{checkpoints}'. Has training run?")
    return available[-1] / PRETRAINED_MODEL_DIR
