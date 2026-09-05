"""Low-rate resource sampler used while rosbag2 is recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
import time

import psutil


running = True


def stop(_signum, _frame):
    global running
    running = False


def directory_size(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def gpu_sample():
    try:
        output = subprocess.run([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"], check=False, capture_output=True,
            text=True, timeout=2).stdout.strip().splitlines()[0]
        utilization, memory = (float(value.strip()) for value in output.split(","))
        return utilization, memory
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None, None


def summary(values):
    clean = [value for value in values if value is not None]
    return {"mean": sum(clean) / len(clean) if clean else None,
            "max": max(clean) if clean else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--bag-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    process = psutil.Process(args.pid)
    process.cpu_percent(None)
    samples, previous_size, previous_time = [], 0, time.monotonic()
    while running:
        now = time.monotonic()
        try:
            current_size = directory_size(args.bag_path) if Path(args.bag_path).exists() else 0
            gpu, vram = gpu_sample()
            samples.append({
                "elapsed_s": now - previous_time if not samples else now - samples[0]["monotonic"],
                "monotonic": now,
                "recorder_cpu_percent": process.cpu_percent(None),
                "recorder_rss_bytes": process.memory_info().rss,
                "system_cpu_percent": psutil.cpu_percent(None),
                "system_memory_percent": psutil.virtual_memory().percent,
                "gpu_utilization_percent": gpu,
                "gpu_memory_mb": vram,
                "bag_size_bytes": current_size,
                "bag_write_bytes_per_second": (
                    max(0, current_size - previous_size) /
                    max(now - previous_time, 1e-6)),
            })
            previous_size, previous_time = current_size, now
        except (psutil.Error, OSError):
            break
        time.sleep(args.interval)
    for sample in samples:
        sample.pop("monotonic", None)
    report = {
        "samples": len(samples),
        "recorder_cpu_percent": summary([s["recorder_cpu_percent"] for s in samples]),
        "recorder_rss_bytes": summary([s["recorder_rss_bytes"] for s in samples]),
        "system_cpu_percent": summary([s["system_cpu_percent"] for s in samples]),
        "system_memory_percent": summary([s["system_memory_percent"] for s in samples]),
        "gpu_utilization_percent": summary([s["gpu_utilization_percent"] for s in samples]),
        "gpu_memory_mb": summary([s["gpu_memory_mb"] for s in samples]),
        "bag_write_bytes_per_second": summary([s["bag_write_bytes_per_second"] for s in samples[1:]]),
        "raw_samples": samples,
    }
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
