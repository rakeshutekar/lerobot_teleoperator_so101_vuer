"""train_remote.sh, driven with stub ssh/rsync so nothing leaves this machine.

The stubs record the exact command string handed to the remote shell, which is the only
place the stage logic is observable. The commands recorded are then fed to LeRobot's own
config parser, so a stage that would die on startup fails here instead.
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "train_remote.sh"

STUB_SSH = """#!/usr/bin/env bash
printf 'SSH\\t%s\\t%s\\n' "$1" "$2" >> "$STUB_LOG"
case "$2" in
  'test -d '*) exit "${STUB_JOB_EXISTS:-1}" ;;
esac
exit 0
"""

STUB_RSYNC = """#!/usr/bin/env bash
printf 'RSYNC\\t%s\\n' "$*" >> "$STUB_LOG"
exit 0
"""

STUB_LOCAL_PY = """#!/usr/bin/env bash
printf 'LOCALPY\\t%s\\n' "$*" >> "$STUB_LOG"
exit 0
"""


class Harness:
    """A copy of the script in a scratch directory, with ssh, rsync and python stubbed."""

    def __init__(self, tmp_path: Path):
        self.home = tmp_path / "repo"
        self.home.mkdir()
        self.script = self.home / "train_remote.sh"
        self.script.write_bytes(SCRIPT.read_bytes())
        self.remote_dir = tmp_path / "remote"
        self.log = tmp_path / "stub.log"
        stubs = tmp_path / "stubs"
        stubs.mkdir()
        for name, body in (("ssh", STUB_SSH), ("rsync", STUB_RSYNC), ("python", STUB_LOCAL_PY)):
            path = stubs / name
            path.write_text(body)
            path.chmod(0o755)
        self.stubs = stubs

    def run(self, stage: str, **env_overrides) -> subprocess.CompletedProcess:
        env = {
            **os.environ,
            "PATH": f"{self.stubs}{os.pathsep}{os.environ['PATH']}",
            "STUB_LOG": str(self.log),
            "GPU_HOST": "user@gpubox",
            "GPU_DIR": str(self.remote_dir),
            "LOCAL_PY": str(self.stubs / "python"),
        }
        env.update({key: str(value) for key, value in env_overrides.items()})
        return subprocess.run(
            ["bash", str(self.script), stage], capture_output=True, text=True, env=env
        )

    def calls(self, kind: str) -> list[str]:
        if not self.log.exists():
            return []
        return [
            line.split("\t", 2)[-1]
            for line in self.log.read_text().splitlines()
            if line.startswith(f"{kind}\t")
        ]

    def train_flags(self) -> list[str]:
        """The lerobot-train flags the remote shell would run, as a real argument list."""
        marker = "-m lerobot.scripts.lerobot_train "
        commands = [call for call in self.calls("SSH") if marker in call]
        assert len(commands) == 1, f"expected one training command, got {commands}"
        return shlex.split(commands[0].split(marker, 1)[1])


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


def build_train_config(flags: list[str]):
    """Parse flags the way lerobot-train does, validate() included, and return the config."""
    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts import lerobot_train  # noqa: F401  registers the policy types

    @parser.wrap()
    def build(cfg: TrainPipelineConfig):
        cfg.validate()
        return cfg

    argv = sys.argv
    sys.argv = ["lerobot-train", *flags]
    try:
        return build()
    finally:
        sys.argv = argv


def test_probe_trains_into_a_throwaway_directory_and_fetches_nothing(harness):
    """A 200-step checkpoint must never end up where pour.py would deploy it from."""
    result = harness.run("probe")

    assert result.returncode == 0, result.stderr
    flags = harness.train_flags()
    assert f"--output_dir={harness.remote_dir}/outputs/train/pour_v1_probe" in flags
    assert "--steps=200" in flags
    assert not [call for call in harness.calls("RSYNC") if "outputs/train" in call]
    assert not (harness.home / "outputs").exists()


def test_validate_starts_the_deployable_run_and_full_continues_it(harness):
    """probe -> validate -> full must run end to end without a FileExistsError."""
    validate = harness.run("validate")
    assert validate.returncode == 0, validate.stderr
    validate_flags = harness.train_flags()
    assert f"--output_dir={harness.remote_dir}/outputs/train/pour_v1" in validate_flags
    assert "--steps=20000" in validate_flags
    assert "--resume=true" not in validate_flags

    harness.log.unlink()
    full = harness.run("full", STUB_JOB_EXISTS=0)
    assert full.returncode == 0, full.stderr
    full_flags = harness.train_flags()
    assert "--resume=true" in full_flags
    assert "--steps=100000" in full_flags
    assert (
        f"--config_path={harness.remote_dir}/outputs/train/pour_v1"
        "/checkpoints/last/pretrained_model/train_config.json" in full_flags
    )


def test_full_starts_a_fresh_run_when_none_exists_yet(harness):
    result = harness.run("full", STUB_JOB_EXISTS=1)

    assert result.returncode == 0, result.stderr
    flags = harness.train_flags()
    assert "--resume=true" not in flags
    assert f"--output_dir={harness.remote_dir}/outputs/train/pour_v1" in flags


def test_the_explicit_resume_override_still_wins_both_ways(harness):
    forced_on = harness.run("full", STUB_JOB_EXISTS=1, RESUME=1)
    assert "--resume=true" in harness.train_flags(), forced_on.stderr

    harness.log.unlink()
    harness.run("full", STUB_JOB_EXISTS=0, RESUME=0)
    assert "--resume=true" not in harness.train_flags()


def test_an_unreachable_host_is_not_read_as_a_missing_run(harness):
    """ssh exit 255 means "no answer", not "no run yet"; starting fresh would be wrong."""
    result = harness.run("full", STUB_JOB_EXISTS=255)

    assert result.returncode == 255
    assert "cannot ask" in result.stderr
    assert not [call for call in harness.calls("SSH") if "lerobot_train" in call]


def test_fetch_pulls_only_the_deployable_run_and_trains_nothing(harness):
    result = harness.run("fetch")

    assert result.returncode == 0, result.stderr
    assert not [call for call in harness.calls("SSH") if "lerobot_train" in call]
    pulls = [call for call in harness.calls("RSYNC") if "outputs/train" in call]
    assert len(pulls) == 1
    assert "pour_v1/" in pulls[0] and "pour_v1_probe" not in pulls[0]


def test_a_fetch_that_brings_back_no_checkpoints_still_succeeds(harness):
    """`ls` on an empty output directory must not fail the script under `set -e`."""
    result = harness.run("fetch")

    assert result.returncode == 0, result.stderr
    assert "(none fetched)" in result.stdout


def test_the_local_stage_uses_the_apple_gpu_and_no_remote_host(harness):
    result = harness.run("local", GPU_HOST="", GPU_DIR="")

    assert result.returncode == 0, result.stderr
    assert harness.calls("SSH") == []
    assert harness.calls("RSYNC") == []
    flags = shlex.split(harness.calls("LOCALPY")[0])
    assert "--policy.device=mps" in flags
    assert "--policy.push_to_hub=false" in flags
    assert f"--output_dir={harness.home}/outputs/train/pour_v1_local" in flags


@pytest.mark.parametrize("stage", ["probe", "validate", "full"])
def test_every_fresh_stage_produces_a_command_lerobot_will_actually_start(harness, stage):
    """push_to_hub defaults to true, and LeRobot then refuses to start without a repo_id."""
    harness.run(stage, STUB_JOB_EXISTS=1)

    config = build_train_config(harness.train_flags())

    assert config.policy.push_to_hub is False


def test_the_resume_command_restarts_the_run_validate_created(harness):
    """The resume flags must load the checkpoint's config and raise its step budget."""
    harness.run("validate")
    started = build_train_config(harness.train_flags())
    checkpoint = Path(started.output_dir) / "checkpoints/last/pretrained_model"
    checkpoint.mkdir(parents=True)
    started.save_pretrained(checkpoint)

    harness.log.unlink()
    harness.run("full", STUB_JOB_EXISTS=0)
    resumed = build_train_config(harness.train_flags())

    assert resumed.resume is True
    assert resumed.steps == 100000
    assert resumed.output_dir == started.output_dir
