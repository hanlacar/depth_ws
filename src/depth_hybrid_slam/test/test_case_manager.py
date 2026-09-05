from depth_hybrid_slam.case_manager_core import (initialize_cases, REQUIRED_METADATA,
                                                 seal_case, sha256, verify_case)
import yaml


def fixture(tmp_path, case="case_1"):
    directory = tmp_path/case
    directory.mkdir()
    (directory/"rtabmap.db").write_bytes(b"fixture-db")
    (directory/"route.csv").write_text("index,x,y\n0,0,0\n", encoding="utf-8")
    route = {"map_id": "map-a", "route_id": "route-a"}
    (directory/"route.yaml").write_text(yaml.safe_dump(route), encoding="utf-8")
    values = {key: "fixture" for key in REQUIRED_METADATA}
    values.update(case_id=case, map_id="map-a", route_id="route-a",
                  route_point_count=1,
                  rtabmap_db_sha256=sha256(directory/"rtabmap.db"),
                  route_sha256=sha256(directory/"route.csv"))
    (directory/"metadata.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")
    (directory/"checksums.sha256").write_text(
        f"{values['rtabmap_db_sha256']}  rtabmap.db\n"
        f"{values['route_sha256']}  route.csv\n", encoding="utf-8")
    return directory


def test_valid_case_and_four_id_boundary(tmp_path):
    fixture(tmp_path)
    valid, problems, metadata = verify_case(tmp_path, "case_1")
    assert valid and not problems and metadata["map_id"] == "map-a"


def test_route_damage_and_map_route_mismatch_are_blocked(tmp_path):
    directory = fixture(tmp_path)
    (directory/"route.csv").write_text("corrupt", encoding="utf-8")
    valid, problems, _ = verify_case(tmp_path, "case_1")
    assert not valid and "ROUTE_CHECKSUM_MISMATCH" in problems
    directory = fixture(tmp_path, "case_2")
    route = yaml.safe_load((directory/"route.yaml").read_text())
    route["map_id"] = "wrong"
    (directory/"route.yaml").write_text(yaml.safe_dump(route), encoding="utf-8")
    assert "MAP_ID_MISMATCH" in verify_case(tmp_path, "case_2")[1]


def test_missing_metadata_is_blocked(tmp_path):
    directory = fixture(tmp_path)
    metadata = yaml.safe_load((directory/"metadata.yaml").read_text())
    del metadata["d456_serial"]
    (directory/"metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    assert "METADATA_MISSING:d456_serial" in verify_case(tmp_path, "case_1")[1]


def test_invalid_case_name_cannot_escape_root(tmp_path):
    try:
        verify_case(tmp_path, "../case_1")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe case id accepted")


def test_initialize_and_seal_are_non_overwriting(tmp_path):
    root = tmp_path/"cases"
    assert len(initialize_cases(root)) == 4
    database = tmp_path/"new.db"
    route_path = tmp_path/"new.csv"
    database.write_bytes(b"sqlite fixture")
    route_path.write_text("index,x,y\n0,0,0\n", encoding="utf-8")
    metadata = {key: "fixture" for key in REQUIRED_METADATA}
    metadata.update(map_id="map-a", route_id="route-a")
    seal_case(root, "case_1", database, route_path, metadata)
    assert verify_case(root, "case_1")[0]
    try:
        seal_case(root, "case_1", database, route_path, metadata)
    except FileExistsError:
        pass
    else:
        raise AssertionError("sealed case was overwritten")
