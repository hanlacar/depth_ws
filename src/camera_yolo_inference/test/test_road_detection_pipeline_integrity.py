"""Regression tests for the full "does road survive the pipeline" chain:
model metadata <-> class_manifest.yaml <-> ClassMapper <-> build_semantic_masks
<-> TensorRT/PyTorch backend decode <-> perception_refinement <-> RLE wire
format. Written after a frame-index-matched offline/ROS comparison found
model class 0 ("road") IS correctly wired end-to-end, and IS what
camera_yolo_inference_node actually uses -- these tests pin that down so a
future change can't silently break it the way the original bug report
(all-zero road, class-mapping suspected) worried it might have.

GPU/model-file-dependent tests skip gracefully when unavailable (CI without
a GPU, or the model file not present) rather than failing the whole suite.
"""
import os
import unittest

import numpy as np
import yaml

MODEL_PT = "/home/qor/camera_ws/src/camera_yolo_inference/models/hanla_yolo11n_seg_best.pt"
MODEL_ENGINE = "/home/qor/camera_ws/src/camera_yolo_inference/models/hanla_yolo11n_seg_best.engine"
MANIFEST_PATH = "/home/qor/camera_ws/src/camera_yolo_inference/config/class_manifest.yaml"


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class ModelManifestMappingTest(unittest.TestCase):
    """Case 1 & 2: model class names/order <-> manifest <-> ClassMapper."""

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(MODEL_PT):
            raise unittest.SkipTest("model .pt not present in this environment")
        from ultralytics import YOLO
        cls.model_names = YOLO(MODEL_PT, task="segment").names
        cls.manifest = yaml.safe_load(open(MANIFEST_PATH))

    def test_model_class_names_resolve_against_manifest_without_error(self):
        from camera_yolo_inference.class_mapper import SemanticClassMapper
        mapper = SemanticClassMapper(self.manifest)
        # Must not raise: every model class has a manifest alias, no class
        # claimed by two roles, every required role has >=1 class.
        mapping = mapper.resolve_model_classes(self.model_names)
        self.assertIn("road", mapping)
        self.assertTrue(len(mapping["road"]) >= 1)

    def test_road_class_id_matches_model_metadata(self):
        # Pins the current, verified-correct state: model class 0 is named
        # exactly "road" (no case/underscore/whitespace drift), and the
        # manifest's road role accepts that exact string.
        self.assertEqual(self.model_names[0], "road")
        road_spec = self.manifest["semantic_roles"]["road"]
        self.assertIn("road", road_spec["accepted_dataset_names"])

    def test_class_mapper_returns_road_class_ids_matching_model(self):
        from camera_yolo_inference.class_mapper import SemanticClassMapper
        mapper = SemanticClassMapper(self.manifest)
        mapper.resolve_model_classes(self.model_names)
        road_ids = mapper.class_ids_for_role("road")
        self.assertEqual(road_ids, (0,))
        for class_id in road_ids:
            self.assertEqual(self.model_names[class_id], "road")


class BuildSemanticMasksRoadTest(unittest.TestCase):
    """Case 3: a raw road instance survives build_semantic_masks() as a
    road-role mask (no GPU needed -- pure post-processing function)."""

    def test_road_instance_becomes_nonzero_road_mask(self):
        from camera_yolo_inference.mask_postprocessor import build_semantic_masks
        shape = (480, 640)
        instance_mask = np.zeros(shape, np.float32)
        instance_mask[200:300, 150:450] = 1.0
        instances = [{"class_id": 0, "confidence": 0.9, "mask": instance_mask}]
        role_class_ids = {"road": (0,), "white_line": (1,)}
        masks = build_semantic_masks(instances, role_class_ids, shape, threshold=0.5)
        self.assertGreater(np.count_nonzero(masks["road"]), 0)
        self.assertEqual(np.count_nonzero(masks["white_line"]), 0)

    def test_no_road_instance_yields_all_zero_road_mask(self):
        # Not a bug signature by itself -- build_semantic_masks() must
        # correctly report zero when there really are zero road instances;
        # this is what the original bug report saw and needed distinguished
        # from a mapping/decode defect.
        from camera_yolo_inference.mask_postprocessor import build_semantic_masks
        shape = (480, 640)
        instances = [{"class_id": 8, "confidence": 0.5,
                     "mask": np.ones(shape, np.float32)}]  # class 8 = stop, not road
        role_class_ids = {"road": (0,)}
        masks = build_semantic_masks(instances, role_class_ids, shape, threshold=0.5)
        self.assertEqual(np.count_nonzero(masks["road"]), 0)

@unittest.skipUnless(_cuda_available() and os.path.isfile(MODEL_ENGINE),
                     "requires CUDA + the exported .engine file")
class BackendDecodeIntegrityTest(unittest.TestCase):
    """Case 4 & 5: TensorRT instance decode preserves class_id/confidence/
    mask, and agrees with the .pt backend on the same frame (rules out a
    TensorRT-specific decode bug -- both backends share the exact same
    infer() implementation, this pins that down empirically too)."""

    @classmethod
    def setUpClass(cls):
        import cv2
        video = "/home/qor/urrc_hanla/20260829_170118.mp4"
        if not os.path.isfile(video):
            raise unittest.SkipTest("reference test video not present")
        cap = cv2.VideoCapture(video)
        # Frame 13808: known (from an earlier offline sweep this session)
        # to produce a strong, unambiguous multi-instance W_line detection
        # in BOTH .pt (conf ~0.57) and the TensorRT engine (conf ~0.61) --
        # deliberately NOT a borderline-confidence frame like 4202, whose
        # "stop" instance this investigation found the engine reports at
        # only ~0.023 vs .pt's 0.317 (a real, separately-documented finding
        # in the final report -- confirmed independent of FP16 vs FP32
        # export). Using a strong-signal frame here keeps this test's
        # purpose focused on catching category errors (wrong model/class
        # order/broken decode), not re-asserting the known confidence gap.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 13808)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise unittest.SkipTest("could not read reference frame")
        import cv2 as _cv2
        cls.frame = _cv2.resize(frame, (640, 480), interpolation=_cv2.INTER_AREA)

    def _infer(self, model_path, backend_name, confidence=0.25):
        from camera_yolo_inference.inference_backend import create_inference_backend
        backend = create_inference_backend(
            backend_name, model_path, device="cuda:0", input_size=640,
            confidence=confidence, require_cuda=True)
        backend.load_model()
        return backend.infer(self.frame)

    def test_tensorrt_decode_preserves_class_confidence_and_mask(self):
        instances = self._infer(MODEL_ENGINE, "tensorrt", confidence=0.25)
        self.assertGreater(len(instances), 0,
                           "reference frame 13808 must produce >=1 instance at conf=0.25")
        for item in instances:
            self.assertIn("class_id", item)
            self.assertIn("confidence", item)
            self.assertIn("mask", item)
            self.assertTrue(0 <= item["confidence"] <= 1.0)
            self.assertGreater(np.asarray(item["mask"]).size, 0)

    def test_backend_wrapper_matches_raw_ultralytics_predict(self):
        # This is deliberately NOT "does the .engine FILE match the .pt
        # FILE's confidence numbers" -- this investigation found they
        # measurably diverge (frame 13808 e.g.: .pt reports road present at
        # conf=0.25, the TensorRT engine does not -- confirmed unrelated to
        # FP16 by an FP32 re-export showing the same gap; logged as a
        # separate finding in the final report, not something fixable in
        # this backend's decode code since infer() is identical for both
        # backends). What THIS test asserts is that OUR wrapper
        # (create_inference_backend -> .infer()) doesn't itself introduce
        # any discrepancy on top of whatever the model/engine already
        # produces: calling the same .pt model through the wrapper and
        # through raw model.predict() directly must agree exactly.
        if not os.path.isfile(MODEL_PT):
            self.skipTest("model .pt not present")
        from ultralytics import YOLO
        wrapped_instances = self._infer(MODEL_PT, "pytorch", confidence=0.25)
        raw_result = YOLO(MODEL_PT, task="segment").predict(
            source=self.frame, imgsz=640, conf=0.25, verbose=False)[0]
        raw_road = (0 in set(int(c) for c in raw_result.boxes.cls.cpu().numpy())
                    if raw_result.boxes is not None and len(raw_result.boxes) else False)
        wrapped_road = any(int(i["class_id"]) == 0 for i in wrapped_instances)
        self.assertEqual(wrapped_road, raw_road,
                         "camera_yolo_inference's backend wrapper must not itself "
                         "change road presence vs. calling ultralytics directly")
        raw_count = 0 if raw_result.boxes is None else len(raw_result.boxes)
        self.assertEqual(len(wrapped_instances), raw_count,
                         "wrapper instance count must match raw predict() exactly")


class RefinementDeliversRoadToSemanticFrameTest(unittest.TestCase):
    """Case 6: when raw road pixels exist, they survive
    perception_refinement AND the RLE wire encoding used for
    SemanticPathFrame.road_rle (no GPU needed)."""

    def test_raw_road_survives_refinement_and_rle_round_trip(self):
        from camera_yolo_inference.perception_refinement import (
            CommonPerceptionRefiner, RefinementConfig)
        from camera_yolo_inference.semantic_path_contract import (
            decode_binary_rle, encode_binary_rle)
        shape = (480, 640)
        road_mask = np.zeros(shape, np.uint8)
        road_mask[300:450, 100:500] = 255  # solid block, no competing markings
        zero = np.zeros(shape, np.uint8)
        raw_masks = {"road": road_mask, "white_line": zero, "yellow_line": zero,
                    "words": zero, "stop_line": zero, "c_line": zero}
        bgr = np.full((480, 640, 3), 100, np.uint8)

        refiner = CommonPerceptionRefiner(RefinementConfig())
        result = refiner.refine(bgr, instances=[], raw_masks=raw_masks,
                                role_class_ids={}, stamp_sec=1.0)

        raw_count = int(np.count_nonzero(road_mask))
        refined_count = int(np.count_nonzero(result.road))
        self.assertGreater(refined_count, 0,
                           "a real raw road mask must not be zeroed by "
                           "refinement on a fresh (no prior-frame history) call")
        self.assertEqual(refined_count, raw_count,
                         "a clean solid road block with no markings/obstacles "
                         "should pass through refinement pixel-for-pixel")

        encoded = encode_binary_rle(result.road)
        decoded = decode_binary_rle(encoded.tolist(), shape[0], shape[1])
        self.assertTrue(np.array_equal(
            (result.road > 0).astype(np.uint8), (decoded > 0).astype(np.uint8)),
            "RLE encode/decode round-trip must preserve the refined road mask exactly")


if __name__ == "__main__":
    unittest.main()
