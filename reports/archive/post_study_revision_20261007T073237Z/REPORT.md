# Small YOLO Segmentation Benchmark — MOTS20

## 1. Experiment Status

PASS WITH WARNINGS

- Models completed: 3/3
- Frames: 2,862 per model; Person GT instances: 26,894
- Run ID: `benchmark-20261005T051531Z`

## 2. Models Tested

| Family | Model | Parameters | GFLOPs | Checkpoint MB |
| --- | --- | --- | --- | --- |
| YOLO26 | YOLO26s-Seg | 11,505,800 | 37.662 | 23.47 |
| YOLO11 | YOLO11s-Seg | 10,113,248 | 33.364 | 20.67 |
| YOLOv8 | YOLOv8s-Seg | 11,821,056 | 40.339 | 23.91 |


## 3. Protocol Compatibility

| Item | Status |
|---|---|
| Dataset | PASS |
| Evaluator | PASS |
| Preprocessing | PASS |
| Input size | PASS |
| Precision | PASS |
| Thresholds | PASS |
| maxDet | PASS |
| Timing protocol | PASS |
| Environment | PASS |

Dataset compatibility: PASS

Preprocessing compatibility: PASS

[Common methodology](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study/blob/main/METHODOLOGY_REFERENCE.md) · [Frozen protocol](EXPERIMENT_PROTOCOL.md)

## 4. Overall Results

| Model | Mask mAP50-95 | AP50 | AP75 | Precision | Recall | F1 | TP-only IoU | TP-only Dice | Inference ms | Pipeline ms | FPS | Peak VRAM MiB | Params | GFLOPs | Checkpoint MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s-Seg | 0.536640 | 0.837980 | 0.586796 | 0.890842 | 0.789581 | 0.837161 | 0.816347 | 0.894970 | 17.308 | 78.828 | 12.686 | 872.67 | 11,505,800 | 37.662 | 23.47 |
| YOLO11s-Seg | 0.483269 | 0.817429 | 0.513823 | 0.868267 | 0.764148 | 0.812887 | 0.793011 | 0.880512 | 15.693 | 88.836 | 11.257 | 1007.50 | 10,113,248 | 33.364 | 20.67 |
| YOLOv8s-Seg | 0.482763 | 0.817024 | 0.506411 | 0.867022 | 0.767309 | 0.814124 | 0.792418 | 0.880020 | 15.220 | 92.173 | 10.849 | 1139.01 | 11,821,056 | 40.339 | 23.91 |


## 5. Tier Winners

| Category | Model | Value |
|---|---|---|
| Highest Mask mAP50-95 | YOLO26s-Seg | 0.536640 |
| Highest AP75 | YOLO26s-Seg | 0.586796 |
| Highest Recall | YOLO26s-Seg | 0.789581 |
| Fastest inference | YOLOv8s-Seg | 15.220 |
| Fastest pipeline | YOLO26s-Seg | 78.828 |
| Highest FPS | YOLO26s-Seg | 12.686 |
| Lowest VRAM | YOLO26s-Seg | 872.67 |

Inference and pipeline latency are in ms/frame; FPS is derived from mean pipeline latency; VRAM is peak allocated MiB.

## 6. Key Findings

- Observation: YOLO26s-Seg has the highest Mask mAP50-95, 0.536640; the gap to the runner-up is 0.053371 on the 0–1 scale.
- Observation: YOLO26s-Seg leads both AP75 and Recall.
- Observation: the fastest forward pass is from YOLOv8s-Seg; the fastest pipeline and highest FPS are from YOLO26s-Seg; the lowest VRAM is from YOLO26s-Seg.
- Closest pair in Mask mAP50-95: YOLO11s-Seg / YOLOv8s-Seg differ by 0.000505 — near-tied descriptively; statistical significance was not tested.
- Interpretation: selection must distinguish accuracy, forward latency, pipeline latency and memory; fewer parameters do not necessarily mean lower latency or VRAM.

## 7. Per-sequence Observations

- YOLO26s-Seg: strongest MOTS20-11 (0.589893); weakest MOTS20-02 (0.419651) by Mask mAP50-95.
- YOLO11s-Seg: strongest MOTS20-05 (0.539178); weakest MOTS20-02 (0.356259) by Mask mAP50-95.
- YOLOv8s-Seg: strongest MOTS20-11 (0.541606); weakest MOTS20-02 (0.359863) by Mask mAP50-95.
- Ranking changes relative to pooled AP: MOTS20-02: YOLO26s-Seg > YOLOv8s-Seg > YOLO11s-Seg; MOTS20-11: YOLO26s-Seg > YOLOv8s-Seg > YOLO11s-Seg

## 8. Efficiency and Resource Observations

- Closest pair in mean inference latency: YOLO11s-Seg / YOLOv8s-Seg differ by 0.473 ms; statistical significance was not tested.
- Closest pair in mean pipeline latency: YOLO11s-Seg / YOLOv8s-Seg differ by 3.337 ms; statistical significance was not tested.

YOLO26s-Seg has the lowest peak allocated VRAM. Loaded/fused parameters, GFLOPs and separate load times are retained in MODEL_COMPLEXITY.csv. Peak reserved VRAM is preserved in [source timing summary](timing/benchmark-20261005T051531Z/clean_repetition/summary.csv). Separate RLE preparation means: yolo26s-seg.pt: 279.956 ms; yolo11s-seg.pt: 330.744 ms; yolov8s-seg.pt: 344.705 ms.

## 9. Warnings and Anomalies

CPU NNPACK unsupported-hardware warnings were captured in this Small run. No pycocotools DeprecationWarning was captured in the Small logs; historical warnings from earlier tiers are not counted as new Small warnings. Evaluator regression passed. No package versions were changed to suppress warnings. Primary timing contains only nine clean runs. Pipeline excludes RLE preparation and disk I/O; it is not end-to-end mask-saving/CCTV throughput.

## 10. Limitations

This evaluates frame-level Person instance segmentation on MOTS20, rather than MOTS tracking. The 26,894 GT instances are frame-level annotations, not unique people. TP-only IoU/Dice are conditional on successful matching.

Consecutive video frames are correlated, and no statistical significance test was performed; small differences are descriptive. Selected qualitative cases do not replace dataset-level metrics.

These measurements do not establish robustness to blur, low light, camera angle or occlusion severity, or deployment suitability. They support candidate selection for later CCTV robustness evaluation only. No weighted score or architectural causal conclusion is used.

## 11. Reproducibility and Source Artifacts

- [TIER_RESULTS.csv](metrics/TIER_RESULTS.csv)
- [PER_SEQUENCE_RESULTS.csv](metrics/PER_SEQUENCE_RESULTS.csv)
- [TIMING_SUMMARY.csv](metrics/TIMING_SUMMARY.csv)
- [MODEL_COMPLEXITY.csv](metrics/MODEL_COMPLEXITY.csv)
- [PREFLIGHT_MAXDET.csv](metrics/PREFLIGHT_MAXDET.csv)

[Standardization provenance](manifests/STANDARDIZATION.json) · [Final integrity](manifests/final_integrity.json) · [Timing source](timing/benchmark-20261005T051531Z/clean_repetition/summary.csv) · [Plots](outputs/plots/INDEX.md)

Lossless per-frame RLE predictions and full telemetry remain local under predictions/benchmark-20261005T051531Z/ and timing/benchmark-20261005T051531Z/. Published manifests record hashes; no inference rerun is required to regenerate metrics.

## 12. Relation to Full Scaling Study

[Master Study](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study) — this is the Small tier only. All five tiers are complete; final 17-model synthesis remains pending and requires explicit authorization.

## Qualitative Analysis

Four same-frame diagnostic comparisons from saved lossless predictions are discussed in [PRESENTATION_SUMMARY_TH.md](PRESENTATION_SUMMARY_TH.md). See [current case selection](outputs/visualizations/qualitative/selection_v2/CASE_SELECTION.md) and [active selection](manifests/QUALITATIVE_SELECTION.json) for selection reasons, shared and tier-specific behaviors, and evidence limits. Comparisons use original MOTS20 frames. No inference or measured values were changed for this documentation update.
