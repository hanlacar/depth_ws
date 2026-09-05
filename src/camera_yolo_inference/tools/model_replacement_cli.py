#!/usr/bin/env python3
import argparse
import json

from camera_yolo_inference.model_replacement import (
    install_model, preflight_model, rollback_model)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    for flag in ("model", "video", "manifest", "output"):
        preflight.add_argument(f"--{flag}", required=True)
    install = commands.add_parser("install")
    install.add_argument("--candidate", required=True)
    install.add_argument("--target", required=True)
    install.add_argument("--timestamp", required=True)
    rollback = commands.add_parser("rollback")
    rollback.add_argument("--backup", required=True)
    rollback.add_argument("--target", required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight_model(args.model, args.video, args.manifest, args.output)
    elif args.command == "install":
        result = install_model(args.candidate, args.target, args.timestamp)
    else:
        result = {"installed_sha256": rollback_model(args.backup, args.target)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
