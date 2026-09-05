"""Common segmentation backend contract for CUDA PyTorch and TensorRT."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
import time
import numpy as np


@dataclass(frozen=True)
class BackendInfo:
    backend: str
    requested_device: str
    actual_device: str
    gpu_name: str
    model_path: str
    model_sha256: str
    task: str
    input_size: int
    confidence_threshold: float
    precision: str


class CudaValidationError(RuntimeError):
    pass


class InferenceBackend(ABC):
    @abstractmethod
    def load_model(self): ...
    @abstractmethod
    def infer(self, image): ...
    @abstractmethod
    def warmup(self, count=3): ...
    @abstractmethod
    def get_model_names(self): ...
    @abstractmethod
    def get_device_info(self): ...
    @abstractmethod
    def get_last_timings(self): ...


class UltralyticsSegmentationBackend(InferenceBackend):
    backend_name = "pytorch"
    expected_suffix = ".pt"
    precision = "fp32"

    def __init__(self, model_path, device="cuda:0", input_size=640,
                 confidence=.25, require_cuda=True, torch_module=None,
                 yolo_factory=None):
        self.model_path = str(model_path)
        self.device = str(device)
        self.input_size = int(input_size)
        self.confidence = float(confidence)
        self.require_cuda = bool(require_cuda)
        self.model = None
        self._torch = torch_module
        self._yolo_factory = yolo_factory
        self._device_index = None
        self._gpu_name = ""
        self._last_timings = {"preprocess_ms": 0.0, "inference_ms": 0.0,
                              "decode_ms": 0.0, "result_transfer_ms": 0.0}

    def _import_torch(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch
        except ImportError as error:
            raise CudaValidationError("PyTorch is not installed") from error
        self._torch = torch
        return torch

    def validate_cuda(self):
        match = re.fullmatch(r"cuda:(\d+)", self.device)
        if self.require_cuda and match is None:
            raise CudaValidationError("CUDA is required; use an explicit device such as cuda:0")
        if match is None:
            return None
        torch = self._import_torch()
        if not torch.cuda.is_available():
            raise CudaValidationError("PyTorch CUDA is unavailable; CPU fallback is forbidden")
        index = int(match.group(1))
        count = int(torch.cuda.device_count())
        if index >= count:
            raise CudaValidationError(
                f"CUDA device index {index} is invalid; available device count is {count}")
        try:
            name = str(torch.cuda.get_device_name(index)).strip()
        except Exception as error:
            raise CudaValidationError(f"cannot query CUDA device {index}") from error
        if not name:
            raise CudaValidationError(f"CUDA device {index} has an empty name")
        self._device_index, self._gpu_name = index, name
        return index

    def validate_model_path(self):
        path = Path(self.model_path)
        if path.suffix.lower() != self.expected_suffix:
            raise ValueError(f"{self.backend_name} model must use {self.expected_suffix}: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"model does not exist: {path}")

    def _validate_runtime(self):
        return None

    def load_model(self):
        self.validate_cuda()
        self.validate_model_path()
        self._validate_runtime()
        if self._yolo_factory is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError("Ultralytics is not installed") from error
            self._yolo_factory = YOLO
        self.model = self._yolo_factory(self.model_path, task="segment")
        self.validate_model_task()
        return self.model

    def validate_model_task(self):
        task = getattr(self.model, "task", None)
        if task != "segment":
            raise ValueError(f"model task must be segment, got {task!r}")

    def get_model_names(self):
        if self.model is None:
            raise RuntimeError("model not loaded")
        return self.model.names

    def infer(self, image):
        if self.model is None:
            raise RuntimeError("model not loaded")
        # CUDA validation imports torch as a side effect, while the supported
        # offline CPU fallback does not. Post-processing needs it in both.
        torch = self._import_torch()
        result = self.model.predict(source=image, imgsz=self.input_size,
                                    conf=self.confidence, device=self.device,
                                    verbose=False)[0]
        speed = getattr(result, "speed", {}) or {}
        self._last_timings = {
            "preprocess_ms": float(speed.get("preprocess", 0.0)),
            "inference_ms": float(speed.get("inference", 0.0)),
            "decode_ms": float(speed.get("postprocess", 0.0)),
            "result_transfer_ms": 0.0,
        }
        if result.masks is None:
            return []
        transfer_started = time.perf_counter()
        mask_tensor = result.masks.data.detach()
        class_tensor = result.boxes.cls.detach().to(dtype=torch.int64)
        unique_classes = torch.unique(class_tensor, sorted=True)
        merged_tensor = torch.stack([
            mask_tensor[class_tensor == class_id].amax(dim=0)
            for class_id in unique_classes
        ])
        merged_masks = merged_tensor.cpu().numpy()
        # boxes.data is [x1, y1, x2, y2, confidence, class].  A single
        # device-to-host transfer avoids three separate CUDA synchronizations
        # for cls / confidence / xyxy on every frame.
        box_data = result.boxes.data.detach().cpu().numpy()
        classes = box_data[:, 5].astype(int, copy=False)
        merged_class_ids = np.unique(classes)
        masks_by_class = {int(class_id): mask for class_id, mask
                          in zip(merged_class_ids, merged_masks)}
        confidences = box_data[:, 4]
        boxes = box_data[:, :4]
        self._last_timings["result_transfer_ms"] = (
            time.perf_counter() - transfer_started) * 1000.0
        return [{"class_id": int(class_id), "confidence": float(confidence),
                 "xyxy": [float(value) for value in box], "mask": mask}
                for class_id, confidence, box, mask
                in zip(classes, confidences, boxes,
                       (masks_by_class[int(class_id)] for class_id in classes))]

    def warmup(self, count=3):
        if int(count) < 0:
            raise ValueError("warmup count must be non-negative")
        image = np.zeros((480, 640, 3), np.uint8)
        for _ in range(int(count)):
            self.infer(image)

    def get_last_timings(self):
        return dict(self._last_timings)

    def get_device_info(self):
        actual = f"cuda:{self._device_index}" if self._device_index is not None else self.device
        model_sha256 = hashlib.sha256(Path(self.model_path).read_bytes()).hexdigest()
        return asdict(BackendInfo(
            backend=self.backend_name, requested_device=self.device,
            actual_device=actual, gpu_name=self._gpu_name,
            model_path=str(Path(self.model_path).resolve()), model_sha256=model_sha256,
            task=str(getattr(self.model, "task", "not_loaded")),
            input_size=self.input_size, confidence_threshold=self.confidence,
            precision=self.precision))


class TensorRTSegmentationBackend(UltralyticsSegmentationBackend):
    backend_name = "tensorrt"
    expected_suffix = ".engine"
    precision = "fp16"

    def _validate_runtime(self):
        try:
            import tensorrt as trt
        except ImportError as error:
            raise RuntimeError("TensorRT Python package is unavailable") from error
        try:
            trt.Runtime(trt.Logger(trt.Logger.ERROR))
        except Exception as error:
            raise RuntimeError("TensorRT runtime initialization failed") from error


def create_inference_backend(backend, model_path, device="cuda:0", input_size=640,
                             confidence=.25, require_cuda=True, **kwargs):
    implementations = {
        "pytorch": UltralyticsSegmentationBackend,
        "tensorrt": TensorRTSegmentationBackend,
    }
    try:
        implementation = implementations[str(backend).strip().lower()]
    except KeyError as error:
        raise ValueError(f"unsupported inference backend: {backend!r}") from error
    return implementation(model_path, device, input_size, confidence, require_cuda, **kwargs)
