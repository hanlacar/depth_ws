from pathlib import Path
import yaml

def load_manifest(path):
    with Path(path).open(encoding="utf-8") as stream:data=yaml.safe_load(stream) or {}
    if data.get("model",{}).get("task")!="segment":raise ValueError("manifest model.task must be segment")
    roles=data.get("semantic_roles",{})
    if "road" not in roles or not roles["road"].get("required",False):raise ValueError("required road semantic role missing")
    claimed={}
    for role,spec in roles.items():
        aliases=spec.get("accepted_dataset_names",[])
        if not aliases or len(aliases)!=len(set(aliases)):raise ValueError(f"invalid aliases for {role}")
        for alias in aliases:
            if alias in claimed:raise ValueError(f"dataset name {alias!r} maps to two semantic roles: {claimed[alias]}, {role}")
            claimed[alias]=role
    return data
