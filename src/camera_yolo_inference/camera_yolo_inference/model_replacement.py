"""Fail-closed helpers for validating and atomically replacing a model."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_copy(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".pending", dir=target.parent)
    os.close(handle)
    temporary = Path(temporary)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if sha256_file(source) != sha256_file(temporary):
            raise IOError("SHA-256 mismatch after candidate copy")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def install_model(candidate, target, timestamp=None):
    candidate, target = Path(candidate).resolve(), Path(target).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    backup = Path(f"{target}.backup_{timestamp}")
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")
    if target.is_file():
        _atomic_copy(target, backup)
    try:
        _atomic_copy(candidate, target)
        if sha256_file(candidate) != sha256_file(target):
            raise IOError("installed model SHA-256 mismatch")
    except Exception:
        if backup.is_file():
            _atomic_copy(backup, target)
        raise
    return {"candidate_sha256": sha256_file(candidate),
            "installed_sha256": sha256_file(target),
            "target": str(target), "backup": str(backup)}


def rollback_model(backup, target):
    backup, target = Path(backup).resolve(), Path(target).resolve()
    if not backup.is_file():
        raise FileNotFoundError(backup)
    _atomic_copy(backup, target)
    if sha256_file(backup) != sha256_file(target):
        raise IOError("rollback SHA-256 mismatch")
    return sha256_file(target)


def preflight_model(model_path, video_path, manifest_path, output_json,
                    frame_count=8, imgsz=640, confidence=0.25,
                    mask_threshold=0.5):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from .class_mapper import SemanticClassMapper
    from .model_manifest import load_manifest

    model_path, video_path = Path(model_path).resolve(), Path(video_path).resolve()
    if not model_path.is_file() or not video_path.is_file():
        raise FileNotFoundError("candidate model or validation video is missing")
    model = YOLO(str(model_path), task="segment")
    if getattr(model, "task", None) != "segment":
        raise ValueError(f"model task must be segment, got {model.task!r}")
    mapper = SemanticClassMapper(load_manifest(manifest_path))
    mapping = mapper.resolve_model_classes(model.names)
    road_ids = set(mapping["road"])
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError("validation video contains no frames")
    indices = np.linspace(0, total-1, max(1, int(frame_count)), dtype=int)
    rows, road_frames, mask_frames = [], 0, 0
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"could not read representative frame {index}")
        result = model.predict(frame, imgsz=int(imgsz), conf=float(confidence),
                               device="cpu", verbose=False)[0]
        if result.masks is not None:
            mask_frames += 1
        classes = ([] if result.boxes is None else
                   [int(value) for value in result.boxes.cls.cpu().tolist()])
        road_pixels = 0
        if result.masks is not None:
            for class_id, mask in zip(classes, result.masks.data.cpu().numpy()):
                if class_id in road_ids:
                    road_pixels += int(np.count_nonzero(mask >= mask_threshold))
        road_frames += int(road_pixels > 0)
        rows.append({"frame": int(index), "classes": classes,
                     "mask_count": 0 if result.masks is None else len(result.masks.data),
                     "road_pixels": road_pixels})
    capture.release()
    if mask_frames == 0:
        raise ValueError("segmentation model produced no masks on representative frames")
    if road_frames == 0:
        raise ValueError("road mask absent from every representative frame")
    report = {"result": "PASS", "model": str(model_path),
              "model_sha256": sha256_file(model_path), "task": model.task,
              "classes": {str(k): v for k, v in dict(model.names).items()},
              "semantic_mapping": mapping, "mapping_warnings": mapper.warnings,
              "video": str(video_path), "video_sha256": sha256_file(video_path),
              "settings": {"imgsz": imgsz, "confidence": confidence,
                           "mask_threshold": mask_threshold},
              "representative_frames": rows, "road_frames": road_frames,
              "mask_frames": mask_frames}
    Path(output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
