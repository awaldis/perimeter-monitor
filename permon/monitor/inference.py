"""
Inference engines: the 'inference' stage, running the YOLO network.

An engine takes the preprocessed image tensor and returns the network's raw
output. Everything hardware-specific about running the model lives here:

  OpenVINOEngine     OpenVINO model folders, run by the OpenVINO runtime on
                     an Intel CPU, integrated/discrete GPU or NPU
  UltralyticsEngine  everything else ultralytics can load (.pt on CPU or
                     NVIDIA CUDA, TensorRT .engine, .onnx, ...)

Input:  float32 array, shape (1, 3, height, width), RGB, values 0-1
Output: torch tensor, shape (1, 4 + classes, candidates). Each candidate
        column is a box (x_center, y_center, width, height in input pixels)
        followed by one score per class. A 480x1600 input gives 15750
        candidates; the 'nms' stage reduces them to the few real detections.
"""

from pathlib import Path

import torch
import yaml


class OpenVINOEngine:
  """Runs an OpenVINO model folder (.xml, .bin, metadata.yaml) with the OpenVINO runtime."""

  def __init__(self, model_dir, device="AUTO"):
    """
    Args:
      model_dir: Folder written by the OpenVINO export
      device: OpenVINO device: "CPU", "GPU" (Intel graphics), "NPU", or
        "AUTO" to let OpenVINO pick the best one present
    """
    import openvino as ov

    model_dir = Path(model_dir)
    core = ov.Core()
    model = core.read_model(str(next(model_dir.glob("*.xml"))))
    # LATENCY: tune for one frame at a time rather than batches
    self.compiled = core.compile_model(model, device.upper(), {"PERFORMANCE_HINT": "LATENCY"})
    self.request = self.compiled.create_infer_request()

    metadata = yaml.safe_load((model_dir / "metadata.yaml").read_text())
    self.names = metadata["names"]
    self.stride = int(metadata.get("stride", 32))
    self.end2end = bool(metadata.get("end2end", False))
    # Exported models have a fixed input size
    self.imgsz = tuple(self.compiled.input().shape[2:])

    devices = ", ".join(self.compiled.get_property("EXECUTION_DEVICES"))
    available = ", ".join(core.available_devices)
    self.description = f"OpenVINO on {devices} (requested {device.upper()}; available: {available})"

  def infer(self, tensor):
    """Run the network on a (1, 3, H, W) float32 array; returns the raw output tensor."""
    return torch.from_numpy(self.request.infer({0: tensor})[0])


class UltralyticsEngine:
  """Runs any model format ultralytics supports through its AutoBackend (PyTorch for .pt)."""

  def __init__(self, model_path, device="AUTO"):
    """
    Args:
      model_path: Model file (.pt, .onnx, .engine, ...)
      device: torch device such as "cpu" or "cuda:0"; "AUTO" uses CUDA if available
    """
    from ultralytics.nn.autobackend import AutoBackend

    if device.upper() == "AUTO":
      device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.device = torch.device(device)
    self.backend = AutoBackend(model=model_path, device=self.device, fp16=False, fuse=True, verbose=False)

    self.names = self.backend.names
    self.stride = int(self.backend.stride)
    self.end2end = bool(getattr(self.backend, "end2end", False))
    self.imgsz = None  # Accepts the size it is given (MODEL_IMGSZ)
    self.description = f"ultralytics AutoBackend ({Path(model_path).suffix or 'folder'}) on {self.device}"

  def infer(self, tensor):
    """Run the network on a (1, 3, H, W) float32 array; returns the raw output tensor."""
    with torch.no_grad():
      output = self.backend(torch.from_numpy(tensor).to(self.device))
    # PyTorch models also return intermediate feature maps; keep the predictions
    return output[0] if isinstance(output, (list, tuple)) else output


def create_engine(model_path, device="AUTO"):
  """Pick the engine for a model: OpenVINO folders get the OpenVINO runtime, the rest ultralytics."""
  path = Path(model_path)
  if path.is_dir() and any(path.glob("*.xml")):
    return OpenVINOEngine(path, device)
  return UltralyticsEngine(str(model_path), device)
