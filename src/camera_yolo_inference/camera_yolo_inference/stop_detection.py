"""Helpers for publishing a compact stop-detection signal."""


def contains_stop(instances, names, minimum_confidence):
    stop_names = {"stop", "stop_line"}
    for item in instances:
        class_id = int(item["class_id"])
        name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        if (
            str(name).strip().lower() in stop_names
            and float(item.get("confidence", 0.0)) >= float(minimum_confidence)
        ):
            return True
    return False
