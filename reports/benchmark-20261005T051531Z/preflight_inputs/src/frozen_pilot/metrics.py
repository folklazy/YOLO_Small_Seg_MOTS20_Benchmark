"""Separate fixed-threshold and COCO confidence-ranked frame-mask metrics."""
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import cached_property
import io
import numpy as np
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mots import encode


@dataclass
class Prediction:
    confidence: float
    class_id: int  # checkpoint class ID, resolved by name; normalized to Person in AP
    bbox_xyxy: list
    mask: np.ndarray

    @cached_property
    def rle(self):
        return encode(self.mask)

    @property
    def shape(self):
        return self.mask.shape

    def compact(self):
        return CompactPrediction(self.confidence,self.class_id,self.bbox_xyxy,self.rle)


@dataclass
class PackedPrediction:
    """Lossless CPU bit-packed native mask; dense array is materialized on demand."""
    confidence: float
    class_id: int
    bbox_xyxy: list
    packed: np.ndarray
    shape: tuple

    @property
    def mask(self):
        return np.unpackbits(self.packed,bitorder='little',count=int(np.prod(self.shape))).reshape(self.shape).view(bool)

    @cached_property
    def rle(self):
        return encode(self.mask)

    def compact(self):
        return CompactPrediction(self.confidence,self.class_id,self.bbox_xyxy,self.rle)


@dataclass
class CompactPrediction:
    """RLE only, retained between frames. Debug mask decoding is lazy/non-cached."""
    confidence: float
    class_id: int
    bbox_xyxy: list
    rle: dict

    @property
    def shape(self):
        return tuple(self.rle['size'])

    @property
    def mask(self):
        return coco_mask.decode(self.rle).view(bool)

    def compact(self):
        return self


def ratios(tp, fp, fn):
    # Undefined precision/recall is NaN (blank/N/A in reports), never a perfect score.
    return {"precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")}


def fixed_metrics(frame, predictions, confidence=0.25, iou_threshold=0.5, ignore_ioa=0.5):
    if not (0 <= confidence <= 1 and 0 < iou_threshold <= 1 and 0 < ignore_ioa <= 1):
        raise ValueError("Invalid metric thresholds")
    if any(p.shape != (frame.height, frame.width) for p in predictions):
        raise ValueError("Prediction/source mask dimensions disagree")
    order = sorted((i for i, p in enumerate(predictions) if p.confidence >= confidence),
                   key=lambda i: (-predictions[i].confidence, i))
    gt = frame.persons
    rles = [predictions[i].rle for i in order]
    ious = (coco_mask.iou(rles, [g.rle for g in gt], [0] * len(gt))
            if rles and gt else np.zeros((len(rles), len(gt))))
    ignore = (coco_mask.iou(rles, [frame.ignore_rle], [1])[:, 0]
              if rles else np.zeros(0))  # crowd denominator = prediction area
    remaining = set(range(len(gt)))
    matches, fp, ignored = [], [], []
    for row, pred_index in enumerate(order):
        candidates = [j for j in remaining if ious[row, j] >= iou_threshold]
        if candidates:
            j = min(candidates, key=lambda j: (-ious[row, j], gt[j].object_id))
            remaining.remove(j)
            iou = float(ious[row, j])
            matches.append({"prediction_index": pred_index, "gt_index": j,
                            "object_id": gt[j].object_id, "iou": iou,
                            "dice": 2 * iou / (1 + iou)})
        elif ignore[row] >= ignore_ioa:
            ignored.append(pred_index)
        else:
            fp.append(pred_index)
    tp, nfp, fn = len(matches), len(fp), len(remaining)
    return {"tp": tp, "fp": nfp, "fn": fn, **ratios(tp, nfp, fn),
            "matched_mask_iou": float(np.mean([m["iou"] for m in matches])) if matches else float("nan"),
            "matched_dice": float(np.mean([m["dice"] for m in matches])) if matches else float("nan"),
            "ignored_predictions": len(ignored), "predictions_at_confidence": len(order),
            "matches": matches, "fp_indices": fp, "fn_indices": sorted(remaining),
            "ignored_indices": ignored}


class MOTSCOCOeval(COCOeval):
    """Standard COCO matching/AP, with a fixed MOTS-region suppression threshold.

    Class-10 regions become one ignored crowd union per frame. For these columns
    only, map prediction-area overlap >= ignore_ioa to 1 (else 0), so suppression
    is constant across AP's IoU thresholds. Valid-person columns are unchanged.
    COCOeval orders nonignored GT first and allows repeated crowd matches.
    """
    def __init__(self, gt, dt, ignore_ioa):
        super().__init__(gt, dt, "segm")
        self.ignore_ioa = ignore_ioa

    def computeIoU(self, imgId, catId):
        ious = super().computeIoU(imgId, catId)
        if isinstance(ious, np.ndarray) and ious.size:
            for j, gt in enumerate(self._gts[imgId, catId]):
                if gt.get("mots_ignore_region"):
                    ious[:, j] = (ious[:, j] >= self.ignore_ioa).astype(float)
        return ious


def ap_metrics(frames, predictions_by_frame, ignore_ioa=0.5, max_dets=100, use_ignore=True, return_details=False):
    if len(frames) != len(predictions_by_frame):
        raise ValueError("Frames and prediction lists must align")
    if not 0 < ignore_ioa <= 1 or not isinstance(max_dets,int) or max_dets<10:
        raise ValueError("Expected ignore IoA in (0,1] and maxDets >= 10")
    images, annotations, detections = [], [], []
    for image_id, (frame, predictions) in enumerate(zip(frames, predictions_by_frame), 1):
        images.append({"id": image_id, "height": frame.height, "width": frame.width})
        for person in frame.persons:
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": 1, "segmentation": person.rle,
                                "area": float(coco_mask.area(person.rle)),
                                "bbox": coco_mask.toBbox(person.rle).tolist(), "iscrowd": 0})
        if use_ignore and coco_mask.area(frame.ignore_rle) > 0:
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": 1, "segmentation": frame.ignore_rle,
                                "area": float(coco_mask.area(frame.ignore_rle)),
                                "bbox": coco_mask.toBbox(frame.ignore_rle).tolist(),
                                "iscrowd": 1, "mots_ignore_region": True})
        for p in predictions:
            if p.shape != (frame.height, frame.width):
                raise ValueError("AP mask/source dimension mismatch")
            detections.append({"image_id": image_id, "category_id": 1,
                               "segmentation": p.rle, "score": p.confidence})
    captured = io.StringIO()
    with redirect_stdout(captured):
        gt = COCO()
        gt.dataset = {"info": {}, "images": images, "annotations": annotations,
                      "categories": [{"id": 1, "name": "Person"}]}
        gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            # COCO.loadRes([]) indexes its first result; build an empty index explicitly.
            dt = COCO()
            dt.dataset = {"images": images, "annotations": [], "categories": gt.dataset["categories"]}
            dt.createIndex()
        evaluator = MOTSCOCOeval(gt, dt, ignore_ioa)
        evaluator.params.imgIds = [x["id"] for x in images]
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 10, max_dets]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    def value(x):
        return float(x) if x >= 0 else float("nan")
    # COCO.summarize hard-codes 100 in stats[0]; read the correct precision slice
    # for diagnostic caps too. Default 100 matches the original stats exactly.
    precision=evaluator.eval['precision'][:,0:101,0,0,-1]
    def mean_valid(a):
        a=a[a>=0]
        return float(a.mean()) if a.size else float('nan')
    result = {"ap50": mean_valid(precision[0]), "ap75": mean_valid(precision[5]),
              "map50_95": mean_valid(precision)}
    all_area = evaluator.params.areaRng[0]
    result["ignored_detections_at_ap50"] = sum(
        int(e["dtIgnore"][0].sum()) for e in evaluator.evalImgs
        if e is not None and e["aRng"] == all_area and e["maxDet"] == max_dets)
    if return_details:
        details=[]
        for e in evaluator.evalImgs:
            if e is None or e['aRng']!=all_area or e['maxDet']!=max_dets:continue
            valid=(e['dtMatches']>0)&~e['dtIgnore']
            details.append({'image_id':int(e['image_id']),
                            'tp_by_iou':valid.sum(axis=1).tolist(),
                            'tail_tp_by_iou':valid[:,100:].sum(axis=1).tolist(),
                            'tail_tp_ranks':[(np.flatnonzero(row[100:])+101).tolist() for row in valid]})
        return result,captured.getvalue(),details
    return result, captured.getvalue()
