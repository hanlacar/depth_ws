import cv2
import numpy as np

def remove_letterbox_padding(mask,raw_shape):
    mask=np.asarray(mask,dtype=np.float32);raw_h,raw_w=raw_shape
    if mask.shape==(raw_h,raw_w):return mask
    h,w=mask.shape;scale=min(w/raw_w,h/raw_h);content_w=max(1,int(round(raw_w*scale)));content_h=max(1,int(round(raw_h*scale)));left=max(0,(w-content_w)//2);top=max(0,(h-content_h)//2)
    cropped=mask[top:top+content_h,left:left+content_w]
    if cropped.size==0:raise ValueError("letterbox crop is empty")
    return cropped
def restore_masks_to_raw_shape(mask,raw_shape):
    cropped=remove_letterbox_padding(mask,raw_shape)
    if cropped.shape==tuple(raw_shape):return cropped
    return cv2.resize(cropped,(raw_shape[1],raw_shape[0]),interpolation=cv2.INTER_LINEAR)
def prepare_instance_masks(instances,raw_shape,threshold):
    prepared=[]
    for instance in instances:
        item=dict(instance)
        item["restored_mask"]=threshold_probability_mask(
            restore_masks_to_raw_shape(instance["mask"],raw_shape),threshold)
        prepared.append(item)
    return prepared
def threshold_probability_mask(mask,threshold):
    array=np.asarray(mask,dtype=np.float32)
    if not np.isfinite(array).all():raise ValueError("mask contains NaN/Inf")
    return array>=threshold
def merge_instances_by_class(instances,class_ids,raw_shape,threshold):
    selected=[instance for instance in instances if int(instance["class_id"]) in class_ids]
    if not selected:return np.zeros(raw_shape,bool)
    restored=[instance.get("restored_mask") for instance in selected]
    if all(mask is not None for mask in restored):
        return np.logical_or.reduce(restored)
    native_shape=np.asarray(selected[0]["mask"]).shape
    if all(np.asarray(instance["mask"]).shape==native_shape for instance in selected):
        native=np.maximum.reduce([np.asarray(instance["mask"],dtype=np.float32) for instance in selected])
        return restore_masks_to_raw_shape(native,raw_shape)>=threshold
    merged=np.zeros(raw_shape,bool)
    for instance in selected:
        merged|=threshold_probability_mask(restore_masks_to_raw_shape(instance["mask"],raw_shape),threshold)
    return merged
def convert_to_mono8(mask):return np.asarray(mask,dtype=np.uint8)*np.uint8(255)


def build_semantic_masks(instances, role_class_ids, raw_shape, threshold):
    """Aggregate all semantic roles with one pass over inference instances."""
    by_class = {}
    for instance in instances:
        by_class.setdefault(int(instance["class_id"]), []).append(instance)
    zero = np.zeros(raw_shape, dtype=np.uint8)
    masks = {}
    for role, class_ids in role_class_ids.items():
        selected = [item for class_id in class_ids
                    for item in by_class.get(int(class_id), ())]
        if not selected:
            masks[role] = zero
            continue
        unique_masks = list({id(item["mask"]): item["mask"]
                             for item in selected}.values())
        native_shape = np.asarray(selected[0]["mask"]).shape
        if all(np.asarray(mask).shape == native_shape for mask in unique_masks):
            if len(unique_masks) == 1:
                native = np.asarray(unique_masks[0], dtype=np.float32)
            else:
                native = np.maximum.reduce(
                    [np.asarray(mask, dtype=np.float32) for mask in unique_masks])
            probability = restore_masks_to_raw_shape(native, raw_shape)
            masks[role] = np.greater_equal(probability, threshold).astype(np.uint8) * 255
        else:
            masks[role] = convert_to_mono8(merge_instances_by_class(
                selected, class_ids, raw_shape, threshold))
    return masks
def validate_output_mask(mask,shape=(480,640),allow_empty=True):
    array=np.asarray(mask)
    if array.shape!=shape or array.dtype!=np.uint8:return False
    if not np.isin(np.unique(array),[0,255]).all():return False
    return allow_empty or np.count_nonzero(array)>0

def has_navigation_mask(masks):
    """Accept a road area or either lane boundary as path evidence."""
    return any(np.count_nonzero(masks.get(role, ()))>0 for role in ("road","white_line","yellow_line"))
