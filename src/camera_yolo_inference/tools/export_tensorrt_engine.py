#!/usr/bin/env python3
"""One-shot RTX-targeted static FP16 TensorRT deployment export."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def nvidia_driver_version():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.splitlines()[0].strip()
    except (FileNotFoundError, IndexError, subprocess.CalledProcessError):
        return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--imgsz",type=int,default=640)
    args=parser.parse_args();source=args.model.resolve();target=args.output.resolve()
    if source.suffix.lower()!=".pt" or not source.is_file():raise ValueError("source must be an existing .pt model")
    if target.suffix.lower()!=".engine":raise ValueError("output must use .engine")
    if args.imgsz!=640:raise ValueError("deployment profile requires imgsz=640")
    import torch,ultralytics,tensorrt as trt
    from ultralytics import YOLO
    if not torch.cuda.is_available() or torch.cuda.device_count()<1:raise RuntimeError("CUDA GPU 0 is unavailable")
    model=YOLO(str(source));
    if model.task!="segment":raise ValueError(f"model task must be segment, got {model.task}")
    # Ultralytics 8.4.x accepts half=True for native TensorRT FP16. The newer
    # quantize=16 path adds an unrelated ModelOpt dependency in this release.
    exported=Path(model.export(format="engine",imgsz=640,half=True,int8=False,batch=1,
                               dynamic=False,device=0,verbose=False)).resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    if exported!=target:shutil.move(str(exported),str(target))
    loaded=YOLO(str(target),task="segment")
    loaded.predict(source=__import__("numpy").zeros((480,640,3),dtype="uint8"),
                   imgsz=640,device="cuda:0",verbose=False)
    metadata={"source_model_filename":source.name,"source_model_sha256":sha256(source),
              "engine_filename":target.name,"ultralytics_version":ultralytics.__version__,
              "tensorrt_version":trt.__version__,"pytorch_version":torch.__version__,
              "torch_cuda_version":torch.version.cuda,"nvidia_driver":nvidia_driver_version(),
              "gpu_name":torch.cuda.get_device_name(0),"compute_capability":list(torch.cuda.get_device_capability(0)),
              "precision":"fp16","batch":1,"imgsz":640,"dynamic":False,"int8":False,
              "task":model.task,"class_count":len(model.names),"class_names":model.names,
              "platform":platform.platform(),"python_version":platform.python_version(),
              "creation_timestamp":datetime.now(timezone.utc).isoformat()}
    metadata_path=target.with_suffix(target.suffix+".json")
    metadata_path.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"engine":str(target),"metadata":str(metadata_path)},indent=2))


if __name__=="__main__":main()
