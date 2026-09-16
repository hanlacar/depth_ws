"""Resolve workspace data without depending on a user name or clone path."""

import os
from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_prefix


def workspace_root(package_name="depth_hybrid_slam"):
    """Return the workspace root for isolated or merged colcon installs."""
    configured = os.environ.get("DEPTH_WS_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    try:
        prefix = Path(get_package_prefix(package_name)).resolve()
    except PackageNotFoundError:
        # Source-tree fallback keeps offline tools/tests portable before the
        # first colcon build: <workspace>/src/<package>/<module>/this_file.py.
        source_workspace = Path(__file__).resolve().parents[3]
        package_xml = source_workspace / "src" / package_name / "package.xml"
        if package_xml.is_file():
            return source_workspace
        raise
    install_directory = prefix.parent if prefix.name == package_name else prefix
    if install_directory.name != "install":
        raise RuntimeError(
            f"cannot derive workspace root from package prefix: {prefix}")
    return install_directory.parent


def workspace_path(*parts):
    return workspace_root().joinpath(*parts)
