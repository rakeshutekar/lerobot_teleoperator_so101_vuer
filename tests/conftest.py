from pathlib import Path

import pytest

# The MuJoCo model lives in the sibling robot-arm repo (hardware-validated by draw_real.py).
SCENE = Path(__file__).resolve().parents[2] / "robot-arm" / "sim" / "so101" / "scene.xml"


@pytest.fixture(scope="session")
def scene_path() -> Path:
    if not SCENE.is_file():
        pytest.skip(f"MuJoCo model not found at {SCENE}")
    return SCENE
