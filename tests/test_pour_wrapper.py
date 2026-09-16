import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest

import pour

REPO = Path(__file__).resolve().parents[1]


def run_wrapper(args, cwd):
    return subprocess.run(
        [sys.executable, str(REPO / "pour.py"), "--print-command", *args],
        capture_output=True, text=True, cwd=cwd,
    )


def robot_config_flags(command: str) -> set[str]:
    """Pull out the printed command's --robot.* flags, cameras arg included whole."""
    segments = re.split(r" (?=--)", command.strip())
    return {segment for segment in segments if segment.startswith("--robot.")}


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


def test_park_failure_is_loud_and_fails_the_run(tmp_path, caplog):
    """A parking failure must be reported and must make the overall run fail."""
    park_script = tmp_path / "fake_park.py"
    park_script.write_text("import sys\nsys.exit(1)\n")

    with caplog.at_level(logging.ERROR, logger="pour"):
        exit_code = pour.run_rollout_and_park(lambda: 0, sys.executable, park_script)

    assert exit_code != 0
    assert any(
        "estop_so101.py" in record.message and "raised" in record.message
        for record in caplog.records
    ), caplog.text


def test_successful_park_does_not_mask_a_rollout_failure(tmp_path):
    """A rollout failure must still fail the run even when parking succeeds."""
    park_script = tmp_path / "fake_park_ok.py"
    park_script.write_text("import sys\nsys.exit(0)\n")

    exit_code = pour.run_rollout_and_park(lambda: 3, sys.executable, park_script)

    assert exit_code != 0


def test_the_arm_is_parked_even_when_the_rollout_raises(tmp_path):
    """Parking is the last line of defence, so it must survive an exception too."""
    parked = tmp_path / "parked"
    park_script = tmp_path / "marker_park.py"
    park_script.write_text(f"from pathlib import Path\nPath({str(parked)!r}).write_text('yes')\n")

    def raising_rollout() -> int:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        pour.run_rollout_and_park(raising_rollout, sys.executable, park_script)

    assert parked.read_text() == "yes"


def test_rollout_runs_in_process_and_passes_its_arguments_through(monkeypatch):
    """In-process is what lets the gain hooks reach the robot lerobot-rollout builds."""
    from lerobot.scripts import lerobot_rollout

    seen = []
    monkeypatch.setattr(sys, "argv", ["pour.py"])
    monkeypatch.setattr(lerobot_rollout, "main", lambda: seen.append(sys.argv[1:]))

    assert pour.run_rollout(["--fps=15", "--duration=60"]) == 0
    assert seen == [["--fps=15", "--duration=60"]]


def test_an_interrupted_rollout_becomes_an_exit_code_so_parking_still_runs(monkeypatch):
    """The rollout used to be a child process that was SIGKILLed before teardown finished."""
    from lerobot.scripts import lerobot_rollout

    def interrupted():
        raise KeyboardInterrupt

    monkeypatch.setattr(sys, "argv", ["pour.py"])
    monkeypatch.setattr(lerobot_rollout, "main", interrupted)

    assert pour.run_rollout([]) == 130


def test_the_deploy_rate_defaults_to_thirty_and_can_be_set(tmp_path):
    """A policy trained at 30 Hz must not be deployed at whatever the rollout defaults to."""
    (tmp_path / "outputs/train/pour_v1/checkpoints/040000/pretrained_model").mkdir(parents=True)
    (tmp_path / "cameras.json").write_text('{"front": 0, "side": 1, "wrist": 2}')

    assert "--fps=30" in run_wrapper([], tmp_path).stdout
    assert "--fps=15" in run_wrapper(["--fps", "15"], tmp_path).stdout


def test_robot_config_and_camera_args_match_between_pour_and_record(tmp_path):
    """pour.py and record_pour.py must run the arm under the identical configuration."""
    (tmp_path / "cameras.json").write_text('{"front": 0, "side": 1, "wrist": 2}')
    run = tmp_path / "outputs/train/pour_v1/checkpoints/040000/pretrained_model"
    run.mkdir(parents=True)

    pour_command = run_wrapper([], tmp_path).stdout
    record_command = subprocess.run(
        [sys.executable, str(REPO / "record_pour.py"), "--print-command"],
        capture_output=True, text=True, cwd=tmp_path,
    ).stdout

    pour_flags = robot_config_flags(pour_command)
    record_flags = robot_config_flags(record_command)
    assert pour_flags, "pour.py printed no --robot.* flags"
    assert pour_flags == record_flags
