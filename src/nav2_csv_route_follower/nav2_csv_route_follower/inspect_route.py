"""Print deterministic JSON analysis of the workspace segmented route files."""
import argparse
import json

from .route_network import RouteNetwork


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", default="/home/qor/depth_ws/routes/network/route_network_segmented.csv")
    parser.add_argument(
        "--yaml", default="/home/qor/depth_ws/routes/network/route_network_segmented.yaml")
    options = parser.parse_args(args)
    network = RouteNetwork.load(options.csv, options.yaml)
    print(json.dumps(network.analysis(), indent=2, ensure_ascii=False))
