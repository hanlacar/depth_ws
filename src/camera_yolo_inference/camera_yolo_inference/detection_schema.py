"""Pure helpers for the JSON detection output."""

import json


def model_name(model_names, class_id):
    if isinstance(model_names, dict):
        return str(model_names.get(class_id, class_id))
    return str(model_names[class_id])


def detection_document(instances, model_names, stamp_sec, stamp_nanosec, frame_id):
    detections = []
    for item in instances:
        class_id = int(item["class_id"])
        detections.append({
            "class_id": class_id, "class_name": model_name(model_names, class_id),
            "confidence": round(float(item.get("confidence", 0.0)), 4),
            "xyxy": [round(float(value), 1) for value in item.get("xyxy", ())],
        })
    return {"timestamp": {"sec": int(stamp_sec), "nanosec": int(stamp_nanosec)},
            "frame_id": str(frame_id), "detections": detections}


def serialize_detection_document(*args, **kwargs):
    return json.dumps(detection_document(*args, **kwargs), separators=(",", ":"))
