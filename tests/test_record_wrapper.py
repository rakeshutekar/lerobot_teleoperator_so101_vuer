import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_record_wrapper_refuses_without_a_camera_map(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO / "record_pour.py"), "--print-command"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "cameras.json" in result.stderr


def test_record_wrapper_command_carries_every_contract_value(tmp_path):
    (tmp_path / "cameras.json").write_text('{"front": 11, "overhead": 22, "wrist": 33}')
    result = subprocess.run(
        [sys.executable, str(REPO / "record_pour.py"), "--print-command"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    command = result.stdout
    for required in (
        "--robot.type=so101_follower",
        "--robot.max_relative_target=3",
        "--teleop.type=so101_vuer",
        "--dataset.repo_id=local/so101_pour_bottle_v1",
        "--dataset.fps=30",
        "--dataset.num_episodes=50",
        "--dataset.episode_time_s=40",
        "--dataset.reset_time_s=15",
        "--dataset.push_to_hub=false",
        "--fps=30",
        "front:", "overhead:", "wrist:",
        "index_or_path: 11", "index_or_path: 22", "index_or_path: 33",
    ):
        assert required in command, f"missing {required}"
