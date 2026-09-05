import unittest


class CameraYoloInferenceImportTest(unittest.TestCase):

    def test_core_modules_import(self):
        from camera_yolo_inference import inference_backend
        from camera_yolo_inference import mask_postprocessor

        self.assertTrue(hasattr(inference_backend, "create_inference_backend"))
        self.assertTrue(hasattr(mask_postprocessor, "build_semantic_masks"))

    def test_cpu_backend_initializes_torch_for_postprocessing(self):
        from camera_yolo_inference.inference_backend import \
            UltralyticsSegmentationBackend
        backend = UltralyticsSegmentationBackend(
            "unused.pt", device="cpu", require_cuda=False)
        self.assertIsNone(backend._torch)
        self.assertIsNotNone(backend._import_torch())
        self.assertIsNotNone(backend._torch)

    def test_detect_only_model_is_rejected(self):
        from camera_yolo_inference.inference_backend import \
            UltralyticsSegmentationBackend
        backend = UltralyticsSegmentationBackend(
            "unused.pt", device="cpu", require_cuda=False)
        backend.model = type("DetectOnly", (), {"task": "detect"})()
        with self.assertRaisesRegex(ValueError, "must be segment"):
            backend.validate_model_task()


if __name__ == "__main__":
    unittest.main()
