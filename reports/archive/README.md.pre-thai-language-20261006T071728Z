# Small YOLO Instance Segmentation Benchmark on MOTS20

## Overview

Pretrained YOLO Person instance segmentation on MOTS20 using the frozen study protocol. Three Small checkpoints were evaluated without training, fine-tuning or adaptation. This is frame-level segmentation, not MOTS tracking.

[PRESENTATION_SUMMARY_TH.md](PRESENTATION_SUMMARY_TH.md) now follows the shared visual-analysis format, with four same-frame MOTS20 cases reconstructed from saved predictions. The numerical summary remains in [RESULTS_SUMMARY_TH.md](RESULTS_SUMMARY_TH.md). No inference was rerun.

## Models

| Family | Model | Tier |
|---|---|---|
| YOLO26 | YOLO26s-Seg | Small (S) |
| YOLO11 | YOLO11s-Seg | Small (S) |
| YOLOv8 | YOLOv8s-Seg | Small (S) |


## Experimental Status

PASS WITH WARNINGS — COMPLETE, run `benchmark-20261005T051531Z`. All three models completed 2,862 frames and three clean timing rounds each.

## Main Result

| Model | Mask mAP50-95 | Recall | F1 | Inference ms | Pipeline ms | FPS | Peak VRAM MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s-Seg | 0.536640 | 0.789581 | 0.837161 | 17.308 | 78.828 | 12.686 | 872.67 |
| YOLO11s-Seg | 0.483269 | 0.764148 | 0.812887 | 15.693 | 88.836 | 11.257 | 1007.50 |
| YOLOv8s-Seg | 0.482763 | 0.767309 | 0.814124 | 15.220 | 92.173 | 10.849 | 1139.01 |


## Reports

- [PRESENTATION_SUMMARY_TH.md](PRESENTATION_SUMMARY_TH.md)
- [RESULTS_SUMMARY_TH.md](RESULTS_SUMMARY_TH.md)
- [REPORT.md](REPORT.md)
- [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)

## Study Navigation

[Largest](https://github.com/folklazy/YOLO_Large_Seg_MOTS20_Benchmark) | [Second-largest](https://github.com/folklazy/YOLO_Second_Largest_Seg_MOTS20_Benchmark) | [Medium](https://github.com/folklazy/YOLO_Medium_Seg_MOTS20_Benchmark) | [Small](https://github.com/folklazy/YOLO_Small_Seg_MOTS20_Benchmark) | [Nano](https://github.com/folklazy/YOLO_Nano_Seg_MOTS20_Benchmark) | [Master Study](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study)

## Reproducibility

[configs/](configs/) · [metrics/](metrics/) · [manifests/](manifests/)
