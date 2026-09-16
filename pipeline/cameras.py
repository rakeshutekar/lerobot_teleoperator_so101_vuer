"""Build LeRobot's --robot.cameras argument from a saved device map.

The three camera names are a contract between recording and deployment: a policy
trained on `wrist`, `front` and `overhead` must be given those same names at rollout.
"""
import json
from pathlib import Path

CAMERA_NAMES: tuple[str, str, str] = ("wrist", "front", "overhead")


def load_camera_map(path: str | Path) -> dict[str, int | str]:
    """Read the {name: device index or path} map written by find_cameras."""
    path = Path(path)
    try:
        camera_map = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"No camera map at '{path}'. Run lerobot-find-cameras and write it, e.g. "
            '{"wrist": 0, "front": 1, "overhead": 2}'
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Camera map '{path}' is not valid JSON: {error}") from error
    except UnicodeDecodeError as error:
        raise ValueError(f"Camera map '{path}' is not valid UTF-8: {error}") from error
    except OSError as error:
        raise OSError(f"Cannot read camera map at '{path}': {error}") from error
    if not isinstance(camera_map, dict):
        raise ValueError(f"Camera map '{path}' must be a JSON object, got {type(camera_map).__name__}")
    return camera_map


def build_cameras_arg(
    camera_map: dict[str, int | str], width: int = 640, height: int = 480, fps: int = 30
) -> str:
    """Render the map as LeRobot's inline camera config string."""
    missing = [name for name in CAMERA_NAMES if name not in camera_map]
    extra = [name for name in camera_map if name not in CAMERA_NAMES]
    if missing or extra:
        raise ValueError(
            f"Camera map must have exactly {list(CAMERA_NAMES)}; missing={missing}, unexpected={extra}"
        )
    entries = ", ".join(
        f"{name}: {{type: opencv, index_or_path: {camera_map[name]}, "
        f"width: {width}, height: {height}, fps: {fps}}}"
        for name in CAMERA_NAMES
    )
    return "{ " + entries + " }"
