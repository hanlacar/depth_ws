"""Explicit post-RViz approval for an already generated map route."""

import argparse

from .map_route_recorder_core import approve_rviz_review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--route", default="")
    parser.add_argument("--map", dest="map_path", default="")
    parser.add_argument(
        "--approve-rviz", action="store_true", required=True,
        help="assert that a human inspected map + route in RViz")
    options = parser.parse_args(argv)
    values = approve_rviz_review(
        options.metadata, options.route, options.map_path)
    print("PASS")
    print(f"metadata={options.metadata}")
    print(f"route_sha256={values['route_csv_sha256']}")
    print(f"rtabmap_db_sha256={values['rtabmap_db_sha256']}")


if __name__ == "__main__":
    main()
