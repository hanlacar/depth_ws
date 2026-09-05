#!/usr/bin/env python3
"""Aggregate a captured /camera/bev/diagnostics stream and locate the exact
pipeline stage where road pixels disappear, frame by frame.

Input: the raw text file produced by

    ros2 topic echo --truncate-length 100000 /camera/bev/diagnostics \\
      > ~/camera_ws/bev_stage_diagnostics.txt

(--truncate-length 100000 matters: ros2 topic echo truncates String fields
at 128 characters by default, which silently cuts these JSON payloads mid
-field and makes them unparseable -- this script warns loudly if it detects
that happened.)

Usage:
    python3 analyze_bev_stage_diagnostics.py [path_to_txt]
    (defaults to ~/camera_ws/bev_stage_diagnostics.txt)
"""
import json
import re
import statistics
import sys
from pathlib import Path

DEFAULT_PATH = Path.home() / "camera_ws" / "bev_stage_diagnostics.txt"


def load_records(path):
    text = path.read_text()
    raw_blocks = re.findall(r"^data: '(.*)'$", text, re.M)
    records, decode_failures = [], 0
    for block in raw_blocks:
        try:
            records.append(json.loads(block))
        except json.JSONDecodeError:
            decode_failures += 1
    return records, decode_failures, len(raw_blocks)


def stage_stats(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return {
        "count": len(clean),
        "mean": statistics.mean(clean),
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
    }


def pct(numerator, denominator):
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def main():
    path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.is_file():
        sys.exit(f"FATAL: file not found: {path}")
    records, decode_failures, raw_block_count = load_records(path)

    print(f"입력 파일: {path}")
    print(f"원시 'data:' 블록 수: {raw_block_count}, JSON 파싱 성공: {len(records)}, "
         f"파싱 실패: {decode_failures}")
    if decode_failures > 0:
        print("  경고: 파싱 실패 블록이 있습니다. ros2 topic echo를 "
             "--truncate-length 100000 없이 실행하면 문자열이 128자에서 "
             "잘려 이런 실패가 발생합니다. 재수집을 권장합니다.")
    if not records:
        sys.exit("FATAL: 파싱된 레코드가 0개입니다. 분석을 중단합니다.")

    total = len(records)
    print(f"\n=== 전체 프레임 수: {total} ===")

    def reasons_of(rec):
        return rec.get("reasons") or []

    input_timeout = sum(1 for r in records if "INPUT_TIMEOUT" in reasons_of(r))
    print(f"\nINPUT_TIMEOUT: {input_timeout}/{total} ({pct(input_timeout, total):.1f}%)")

    stages = ["raw_road_pixels", "refined_road_pixels", "decoded_road_pixels",
             "projected_road_pixels", "ego_component_pixels", "safe_road_pixels"]
    stage_labels = {"raw_road_pixels": "raw", "refined_road_pixels": "refined",
                    "decoded_road_pixels": "decoded",
                    "projected_road_pixels": "projected",
                    "ego_component_pixels": "ego_component",
                    "safe_road_pixels": "safe_road"}

    # Only frames carrying stage data at all (i.e. not INPUT_TIMEOUT/
    # CALIBRATION_INVALID/etc., where every stage is null by design).
    with_stage_data = [r for r in records if r.get("raw_road_pixels") is not None]
    print(f"\n단계별 픽셀 데이터를 가진 프레임: {len(with_stage_data)}/{total} "
         f"({pct(len(with_stage_data), total):.1f}%) "
         f"(나머지는 INPUT_TIMEOUT/CALIBRATION_INVALID 등 -- 정상적으로 null)")

    print("\n=== 단계 경계 드롭 지점 (분자/분모는 단계별 픽셀 데이터를 가진 프레임 기준) ===")
    n = len(with_stage_data)

    def between(lo_key, hi_key, lo_positive, hi_zero):
        count = sum(1 for r in with_stage_data
                    if lo_positive(r.get(lo_key)) and hi_zero(r.get(hi_key)))
        return count

    raw_zero = sum(1 for r in with_stage_data if (r.get("raw_road_pixels") or 0) == 0)
    print(f"raw road = 0: {raw_zero}/{n} ({pct(raw_zero, n):.1f}%)  "
         "-- YOLO가 이 프레임에서 road를 아예 검출 못함")

    raw_pos_refined_zero = between("raw_road_pixels", "refined_road_pixels",
                                    lambda v: (v or 0) > 0, lambda v: (v or 0) == 0)
    print(f"raw > 0, refined = 0: {raw_pos_refined_zero}/{n} "
         f"({pct(raw_pos_refined_zero, n):.1f}%)  -- perception_refinement 단계에서 사라짐")

    refined_pos_projected_zero = between(
        "refined_road_pixels", "projected_road_pixels",
        lambda v: (v or 0) > 0, lambda v: (v or 0) == 0)
    print(f"refined > 0, projected = 0: {refined_pos_projected_zero}/{n} "
         f"({pct(refined_pos_projected_zero, n):.1f}%)  "
         "-- RLE 디코드~BEV 투영 사이에서 사라짐 (decoded도 함께 확인 권장)")

    projected_pos_ego_zero = between(
        "projected_road_pixels", "ego_component_pixels",
        lambda v: (v or 0) > 0, lambda v: (v or 0) == 0)
    print(f"projected > 0, ego component = 0: {projected_pos_ego_zero}/{n} "
         f"({pct(projected_pos_ego_zero, n):.1f}%)  "
         "-- road는 있지만 자차와 연결된 컴포넌트가 없음 (EGO_ROAD_MISSING 원인)")

    ego_pos_safe_zero = between(
        "ego_component_pixels", "safe_road_pixels",
        lambda v: (v or 0) > 0, lambda v: (v or 0) == 0)
    print(f"ego component > 0, safe road = 0: {ego_pos_safe_zero}/{n} "
         f"({pct(ego_pos_safe_zero, n):.1f}%)  "
         "-- 연결은 됐지만 차량 폭+여유 폭을 만족하는 안전 영역이 없음")

    print("\n=== VALID / DEGRADED / INVALID (planner_state 기준) ===")
    state_counts = {}
    for r in records:
        state = r.get("planner_state") or r.get("state") or "(missing)"
        state_counts[state] = state_counts.get(state, 0) + 1
    for state, count in sorted(state_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {state:<10} {count:>4}  ({pct(count, total):.1f}%)")

    print("\n=== 단계별 road 픽셀 통계 (null 제외, 값이 있는 프레임만) ===")
    print(f"{'stage':<16}{'count':<8}{'mean':<12}{'median':<12}{'min':<10}{'max':<10}")
    for key in stages:
        stats = stage_stats([r.get(key) for r in records])
        label = stage_labels[key]
        if stats is None:
            print(f"{label:<16}(값 있는 프레임 없음)")
        else:
            print(f"{label:<16}{stats['count']:<8}{stats['mean']:<12.1f}"
                 f"{stats['median']:<12.1f}{stats['min']:<10}{stats['max']:<10}")

    print("\n=== semantic FPS 평균 ===")
    fps_values = [r.get("semantic_input_fps") for r in records
                 if r.get("semantic_input_fps") is not None]
    if fps_values:
        print(f"  mean={statistics.mean(fps_values):.2f}  "
             f"(n={len(fps_values)}, min={min(fps_values):.2f}, max={max(fps_values):.2f})")
    else:
        print("  값 없음")

    print("\n=== end-to-end latency (ms) ===")
    lat_values = sorted(r.get("end_to_end_latency_ms") for r in records
                        if r.get("end_to_end_latency_ms") is not None)
    if lat_values:
        p95_index = min(len(lat_values) - 1, int(round(0.95 * (len(lat_values) - 1))))
        print(f"  mean={statistics.mean(lat_values):.2f}  p95={lat_values[p95_index]:.2f}  "
             f"(n={len(lat_values)}, min={lat_values[0]:.2f}, max={lat_values[-1]:.2f})")
    else:
        print("  값 없음 (전부 INPUT_TIMEOUT/CALIBRATION_INVALID 등이었을 수 있음)")

    print("\n=== stamp_ns 일관성 확인 (VALID/DEGRADED 프레임 중 표본) ===")
    consistent_samples = [r for r in records
                          if r.get("planner_state") in ("VALID", "DEGRADED")
                          and r.get("raw_road_pixels") is not None][:5]
    if consistent_samples:
        for r in consistent_samples:
            print(f"  stamp_ns={r.get('stamp_ns')} source_stamp_ns={r.get('source_stamp_ns')} "
                 f"-> 동일 프레임 기준 raw={r.get('raw_road_pixels')} "
                 f"refined={r.get('refined_road_pixels')} decoded={r.get('decoded_road_pixels')} "
                 f"projected={r.get('projected_road_pixels')} "
                 f"ego={r.get('ego_component_pixels')} safe={r.get('safe_road_pixels')}")
        all_match = all(r.get("stamp_ns") == r.get("source_stamp_ns")
                        for r in consistent_samples)
        print(f"  stamp_ns == source_stamp_ns 전부 일치: {all_match}")
    else:
        print("  VALID/DEGRADED 표본 없음")


if __name__ == "__main__":
    main()
