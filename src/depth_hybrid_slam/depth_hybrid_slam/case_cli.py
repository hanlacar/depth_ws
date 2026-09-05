"""Read-only CLI for selecting and verifying a map/route case."""

import argparse
import json

import yaml

from .case_manager_core import initialize_cases, seal_case, verify_case


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("case_id", choices=[f"case_{i}" for i in range(1, 5)])
    verify.add_argument("--root", default="maps")
    initialize = commands.add_parser("init")
    initialize.add_argument("--root", default="maps")
    seal = commands.add_parser("seal")
    seal.add_argument("case_id", choices=[f"case_{i}" for i in range(1, 5)])
    seal.add_argument("--root", default="maps")
    seal.add_argument("--db", required=True)
    seal.add_argument("--route", required=True)
    seal.add_argument("--metadata", required=True)
    args = parser.parse_args()
    if args.command == "init":
        created = initialize_cases(args.root)
        print(json.dumps({"directories": [str(path) for path in created]}, indent=2))
        return
    if args.command == "seal":
        with open(args.metadata, encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream) or {}
        directory = seal_case(args.root, args.case_id, args.db, args.route, metadata)
        print(json.dumps({"sealed": str(directory)}, ensure_ascii=False, indent=2))
        return
    valid, problems, metadata = verify_case(args.root, args.case_id)
    print(json.dumps({"verified": valid, "problems": problems,
                      "metadata": metadata}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if valid else 2)
