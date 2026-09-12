import importlib.util
from pathlib import Path


def test_self_contained_audit_passes_source_tree():
    workspace = Path(__file__).parents[3]
    script = workspace/"tools"/"audit_depth_ws_self_contained.py"
    spec = importlib.util.spec_from_file_location("depth_ws_audit", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.audit(require_installed=False)
    assert result["pass"], result
