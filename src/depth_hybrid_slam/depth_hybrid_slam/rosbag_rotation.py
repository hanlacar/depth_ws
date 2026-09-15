"""Safe timestamp naming and completed-bag rotation for production launch."""

from datetime import datetime
from pathlib import Path
import re
import shutil


RUN_PATTERN = re.compile(r"^run_(\d{8}_\d{6})$")


def run_name(now=None):
    return "run_"+(now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def completed_bags(root):
    root = Path(root).expanduser().resolve()
    if not root.exists():
        return ()
    values = []
    for path in root.iterdir():
        match = RUN_PATTERN.fullmatch(path.name)
        if (path.is_dir() and match and (path/"metadata.yaml").is_file() and
                not (path/".active").exists()):
            values.append((match.group(1), path))
    return tuple(path for _, path in sorted(values))


def rotate_completed_bags(root, keep_completed):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    keep = max(0, int(keep_completed))
    values = list(completed_bags(root))
    while len(values) > keep:
        target = values.pop(0).resolve()
        if target.parent != root or not RUN_PATTERN.fullmatch(target.name):
            raise RuntimeError("refusing to remove a path outside rosbag root")
        shutil.rmtree(target)
    return tuple(values)


def prepare_bag_path(root, now=None, max_bags=3):
    root = Path(root).expanduser().resolve()
    maximum = int(max_bags)
    if maximum < 1:
        raise ValueError("max_bags must be positive")
    rotate_completed_bags(root, maximum-1)
    output = root/run_name(now)
    if output.exists():
        raise RuntimeError(f"rosbag output already exists: {output}")
    return output
