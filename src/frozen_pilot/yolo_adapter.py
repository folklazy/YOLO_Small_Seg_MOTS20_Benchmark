"""Ultralytics segmentation, native-resolution masks and synchronized stage timing."""
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import torch
from metrics import PackedPrediction


def pack_binary_masks(masks):
    """Lossless little-endian bit packing without an int64 N*H*W temporary."""
    if masks.dtype != torch.uint8 or masks.ndim != 3:
        raise ValueError('Expected native binary uint8 NxHxW masks')
    if masks.shape[0]==0:
        return masks.new_empty((0,(masks.shape[1]*masks.shape[2]+7)//8))
    flat=masks.reshape(masks.shape[0],-1)
    padding=(-flat.shape[1])%8
    if padding:
        flat=torch.nn.functional.pad(flat,(0,padding))
    packed=flat[:,0::8].clone()
    for bit in range(1,8):
        packed.bitwise_or_(flat[:,bit::8] << bit)
    return packed


class YOLOAdapter:
    def __init__(self, checkpoint, config, framework_dir):
        from ultralytics import YOLO
        from ultralytics.models.yolo.segment.predict import SegmentationPredictor
        from ultralytics.utils import DEFAULT_CFG_DICT
        from ultralytics.utils.checks import check_imgsz
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Local checkpoint required; downloading disabled: {checkpoint}")
        model = YOLO(str(checkpoint), task="segment")
        if model.task != "segment":
            raise ValueError("A segmentation checkpoint is required")
        persons = [i for i, name in model.names.items() if name.lower() == "person"]
        if len(persons) != 1:
            raise ValueError("Cannot resolve exactly one Person class from checkpoint")
        self.person_class = persons[0]
        overrides = dict(task="segment", mode="predict", device=config["device"],
                         imgsz=config["imgsz"], conf=config["ap_confidence_floor"],
                         iou=config["nms_iou"], max_det=config["max_detections"],
                         classes=[self.person_class], retina_masks=True, rect=False,
                         save=False, save_txt=False, verbose=False, visualize=False,
                         project=str(framework_dir), name="predict", exist_ok=False)
        # Current Ultralytics uses quantize; older compatible versions use half.
        if "quantize" in DEFAULT_CFG_DICT:
            overrides["quantize"] = 16 if config["precision"] == "fp16" else 32
        else:
            overrides["half"] = config["precision"] == "fp16"
        self.predictor = SegmentationPredictor(overrides=overrides)
        self.predictor.setup_model(model=model.model, verbose=False)
        self.predictor.imgsz = check_imgsz(config["imgsz"], stride=self.predictor.model.stride, min_dim=2)
        self.predictor.source_type = SimpleNamespace(tensor=False, from_img=True, stream=False)
        self.device = self.predictor.device
        self.precision = "fp16" if self.predictor.model.fp16 else "fp32"
        if self.precision != config["precision"]:
            raise RuntimeError("Requested and actual model precision disagree")

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict(self, image, path):
        p = self.predictor
        p.batch = ([str(path)], [image], [""])
        self.sync()
        start = perf_counter()
        tensor = p.preprocess([image])
        self.sync()
        pre_end = perf_counter()
        raw = p.inference(tensor)
        self.sync()
        infer_end = perf_counter()
        result = p.postprocess(raw, tensor, [image])[0]
        self.sync()
        native_end=perf_counter()
        predictions = []
        stage={'ultralytics_postprocess_inclusive_ms':(native_end-infer_end)*1000,
               'binary_validation_gpu_ms':0.,'mask_bitpack_gpu_ms':0.,
               'packed_mask_gpu_to_cpu_ms':0.,'mask_numpy_view_ms':0.,
               'bbox_gpu_to_cpu_ms':0.,'bbox_numpy_view_ms':0.,
               'prediction_objects_and_bbox_extract_ms':0.}
        if len(result.boxes):
            if result.masks is None:
                raise RuntimeError("Detections without segmentation masks")
            masks=result.masks.data
            if len(masks)!=len(result.boxes) or tuple(masks.shape[1:])!=image.shape[:2]:
                raise RuntimeError("Native mask/source dimensions disagree")
            start_validation=perf_counter()
            if masks.dtype!=torch.uint8 or int(masks.max().item())>1:
                raise RuntimeError("Expected binary native-resolution masks")
            self.sync();validated=perf_counter()
            stage['binary_validation_gpu_ms']=(validated-start_validation)*1000
            packed=pack_binary_masks(masks)
            self.sync();packed_end=perf_counter()
            stage['mask_bitpack_gpu_ms']=(packed_end-validated)*1000
            cpu_packed=packed.cpu()
            self.sync();transfer_end=perf_counter()
            stage['packed_mask_gpu_to_cpu_ms']=(transfer_end-packed_end)*1000
            bits=cpu_packed.numpy()
            view_end=perf_counter();stage['mask_numpy_view_ms']=(view_end-transfer_end)*1000
            cpu_boxes=result.boxes.data.cpu()
            self.sync();box_end=perf_counter();stage['bbox_gpu_to_cpu_ms']=(box_end-view_end)*1000
            boxes=cpu_boxes.numpy()
            box_view_end=perf_counter();stage['bbox_numpy_view_ms']=(box_view_end-box_end)*1000
            for box, mask_bits in zip(boxes, bits):
                if int(box[5]) != self.person_class:
                    raise RuntimeError("Non-Person prediction escaped filtering")
                predictions.append(PackedPrediction(float(box[4]),int(box[5]),box[:4].tolist(),mask_bits,image.shape[:2]))
            stage['prediction_objects_and_bbox_extract_ms']=(perf_counter()-box_view_end)*1000
        self.sync()
        end = perf_counter()
        return predictions, {"preprocess_ms": (pre_end-start)*1000,
                             "inference_ms": (infer_end-pre_end)*1000,
                             "postprocess_ms": (end-infer_end)*1000,
                             "total_ms": (end-start)*1000,
                             "fps": 1/(end-start),
                             "image_height": image.shape[0], "image_width": image.shape[1],
                             "input_height": tensor.shape[2], "input_width": tensor.shape[3],**stage}

    def warmup(self, image, path, iterations):
        for _ in range(iterations):
            self.predict(image, path)
        self.sync()

    def reset_peak_memory(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory(self):
        if self.device.type != "cuda":
            return {"peak_gpu_allocated_mib": None, "peak_gpu_reserved_mib": None}
        return {"peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(self.device)/1024**2,
                "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(self.device)/1024**2}
