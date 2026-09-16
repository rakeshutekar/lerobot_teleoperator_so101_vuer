#!/usr/bin/env python3
"""Resolve camera UIDs to OpenCV indices and write cameras.json before recording.

Why this exists: record_pour.py reads cameras.json as {name: index}. On macOS the
index of a UVC camera is its position in AVFoundation's device list, which changes
whenever USB topology changes. Two of the three cameras are identical InnoMakers, so
a swap between "wrist" and "front" produces no error — just a dataset where every
episode has the views crossed. Nothing downstream can detect that.

The teleop path (camera_views.py) already resolves by AVFoundation uniqueID. This
does the same for recording, using identical enumeration so the two agree.

    ./preflight_cameras.py                 # reads camera_uids.json, writes cameras.json
    ./preflight_cameras.py --check         # verify only, exit 1 on any problem
    ./preflight_cameras.py --show          # list every camera with its UID and index

camera_uids.json is the same map run_quest.py takes as --teleop.camera_uids:

    {"wrist": "0x1200000c456366", "front": "0x1300000c456366", "overhead": "0x21300000c456300"}

Roles are wrist/front/overhead, matching pipeline.cameras.CAMERA_NAMES exactly.
"""
import argparse
import json
import sys
from pathlib import Path

ROLES = ("wrist", "front", "overhead")  # == pipeline.cameras.CAMERA_NAMES


def enumerate_cameras():
    """Same enumeration as camera_views.camera_index: video + muxed, sorted by UID."""
    try:
        import AVFoundation as AV
    except ImportError:
        sys.exit("PyObjC AVFoundation not importable — run this from the repo .venv on the Mac.")
    devices = sorted(
        list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))
        + list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed)),
        key=lambda d: str(d.uniqueID()),
    )
    return [
        {
            "index": i,
            "uid": str(d.uniqueID()),
            "name": str(d.localizedName()),
            "model": str(d.modelID()),
            "busy": bool(d.isInUseByAnotherApplication()),
        }
        for i, d in enumerate(devices)
    ]


def resolve(uids: dict, devices: list) -> dict:
    """Map each role's UID to its current index. Every failure names the role."""
    problems, out = [], {}
    if set(uids) != set(ROLES):
        problems.append(f"camera_uids.json must have exactly {list(ROLES)}, got {sorted(uids)}")
    if len(set(uids.values())) != len(uids):
        problems.append("two roles share a UID")
    by_uid = {d["uid"]: d for d in devices}
    for role in ROLES:
        uid = uids.get(role)
        if uid is None:
            continue
        dev = by_uid.get(uid)
        if dev is None:
            problems.append(f"{role}: UID {uid} not connected")
            continue
        if not dev["model"].startswith("UVC Camera"):
            problems.append(f"{role}: {uid} is not an external UVC camera ({dev['model']})")
        if dev["busy"]:
            problems.append(f"{role}: {uid} is in use by another application")
        out[role] = dev["index"]
    if problems:
        raise RuntimeError("\n".join(problems))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", default="camera_uids.json")
    ap.add_argument("--out", default="cameras.json")
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    ap.add_argument("--show", action="store_true", help="list connected cameras and exit")
    a = ap.parse_args()

    devices = enumerate_cameras()
    if a.show:
        for d in devices:
            flag = " (busy)" if d["busy"] else ""
            print(f"  [{d['index']}] {d['uid']:<22} {d['name']}  {d['model']}{flag}")
        return 0

    try:
        uids = json.loads(Path(a.uids).read_text())
    except FileNotFoundError:
        print(f"no {a.uids}; run with --show and write one", file=sys.stderr)
        return 2

    try:
        mapping = resolve(uids, devices)
    except RuntimeError as e:
        print(f"camera pre-flight FAILED:\n{e}", file=sys.stderr)
        return 1

    for role in ROLES:
        print(f"  {role:<9} index {mapping[role]}   {uids[role]}")

    if a.check:
        print("ok (nothing written)")
        return 0

    out = Path(a.out)
    old = json.loads(out.read_text()) if out.exists() else None
    out.write_text(json.dumps(mapping, indent=2) + "\n")
    if old is not None and old != mapping:
        print(f"NOTE: {out} changed from {old} — indices moved since last run")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
