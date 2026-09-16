import json
import os

import pytest

from pipeline.cameras import CAMERA_NAMES, build_cameras_arg, load_camera_map


def test_camera_names_are_the_contract():
    assert CAMERA_NAMES == ("front", "side", "wrist")


def test_build_cameras_arg_emits_every_camera_with_format():
    arg = build_cameras_arg({"front": 0, "side": 1, "wrist": 2})
    for name, index in (("front", 0), ("side", 1), ("wrist", 2)):
        assert f"{name}: {{type: opencv, index_or_path: {index}, width: 640, height: 480, fps: 30}}" in arg
    assert arg.startswith("{") and arg.endswith("}")


def test_build_cameras_arg_accepts_device_paths():
    arg = build_cameras_arg({"front": "/dev/video0", "side": 1, "wrist": 2})
    assert "index_or_path: /dev/video0" in arg


def test_build_cameras_arg_rejects_wrong_names():
    with pytest.raises(ValueError, match="wrist"):
        build_cameras_arg({"front": 0, "side": 1})
    with pytest.raises(ValueError, match="overhead"):
        build_cameras_arg({"front": 0, "side": 1, "wrist": 2, "overhead": 3})


def test_load_camera_map_reads_json(tmp_path):
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps({"front": 0, "side": 1, "wrist": 2}))
    assert load_camera_map(path) == {"front": 0, "side": 1, "wrist": 2}


def test_load_camera_map_error_names_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="cameras.json"):
        load_camera_map(tmp_path / "cameras.json")


def test_load_camera_map_rejects_non_dict_json(tmp_path):
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps([0, 1, 2]))
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_camera_map(path)


def test_load_camera_map_rejects_unreadable_file(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("Running as root, cannot test permission denied")
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps({"front": 0, "side": 1, "wrist": 2}))
    path.chmod(0o000)
    try:
        with pytest.raises(OSError, match="Cannot read camera map"):
            load_camera_map(path)
    finally:
        path.chmod(0o644)


def test_load_camera_map_rejects_directory(tmp_path):
    with pytest.raises(OSError, match="Cannot read camera map"):
        load_camera_map(tmp_path)


def test_load_camera_map_rejects_non_utf8(tmp_path):
    path = tmp_path / "cameras.json"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        load_camera_map(path)
