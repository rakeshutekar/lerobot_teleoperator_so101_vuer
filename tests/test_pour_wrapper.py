import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_wrapper(args, cwd):
    return subprocess.run(
        [sys.executable, str(REPO / "pour.py"), "--print-command", *args],
        capture_output=True, text=True, cwd=cwd,
    )


def test_command_uses_the_resolved_checkpoint_and_contract_values(tmp_path):
    run = tmp_path / "outputs/train/pour_v1/checkpoints/040000/pretrained_model"
    run.mkdir(parents=True)
    (tmp_path / "cameras.json").write_text('{"front": 0, "side": 1, "wrist": 2}')
    result = run_wrapper([], tmp_path)
    assert result.returncode == 0, result.stderr
    command = result.stdout
    for required in (
        "--strategy.type=base",
        "040000/pretrained_model",
        "--robot.type=so101_follower",
        "--robot.max_relative_target=3",
        "--task=pour from the bottle into the cup",
        "--duration=",
        "front:", "side:", "wrist:",
    ):
        assert required in command, f"missing {required}"


def test_explicit_checkpoint_is_honoured(tmp_path):
    for step in ("020000", "040000"):
        (tmp_path / f"outputs/train/pour_v1/checkpoints/{step}/pretrained_model").mkdir(parents=True)
    (tmp_path / "cameras.json").write_text('{"front": 0, "side": 1, "wrist": 2}')
    result = run_wrapper(["--checkpoint", "020000"], tmp_path)
    assert "020000/pretrained_model" in result.stdout
    assert "040000" not in result.stdout


def test_missing_training_output_fails_with_a_clear_message(tmp_path):
    (tmp_path / "cameras.json").write_text('{"front": 0, "side": 1, "wrist": 2}')
    result = run_wrapper([], tmp_path)
    assert result.returncode != 0
    assert "checkpoints" in result.stderr.lower()
