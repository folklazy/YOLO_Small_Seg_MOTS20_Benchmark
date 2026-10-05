# Small (S) YOLO segmentation protocol — frozen benchmark

This Small-only run is explicitly authorized by the user. Exactly three official pretrained
checkpoints: `yolo26s-seg.pt`, `yolo11s-seg.pt`, `yolov8s-seg.pt`. No training/fine-tuning/adaptation.
Only missing official checkpoints are downloaded; no package changes.

## Common scientific protocol

Inherit STUDY_STANDARD.md and BASELINE_REFERENCE.json verbatim: MOTS20 train sequences
02/05/09/11; 2,862 ordered frames / 26,894 Person GT annotations; GT class 2 / ignore class 10.
Exact baseline ordered image, 100-frame preflight, 100-frame timing and visualization manifests.
Frozen aspect-preserving letterbox preprocessing, RGB BCHW FP32 /255, batch 1, 640×640,
CUDA:0 respecting device visibility, retina_masks=True, rect=False, augment=False,
nms=True, backend end2end=False, Person model class 0, agnostic_nms=False.
AP confidence floor 0.001; fixed confidence >=0.25; NMS box IoU 0.70;
model max_det=1000 with cap/NMS truncation stop gates. Frozen one-to-one mask IoU >=0.50,
valid GT priority, unmatched ignore IOA >=0.50. Person-only pooled COCO-style mask AP,
IoU .50:.05:.95, 101 recall points; common AP maxDet=200.
Preflight evaluates 100/200/300/1000, reference 1000, strict absolute differences <0.0001
for AP50/AP75/mAP. STOP before full accuracy if 200 fails.

## Execution and timing

Sequential accuracy order YOLO26s → YOLO11s → YOLOv8s, lossless native-resolution RLE saved.
Shared validation once; frozen evaluator regression before preflight.
Frozen seeded/cyclic timing family order retained; 100 ordered frames, 10 warmups,
three clean rounds/model. CUDA synchronize at stage boundaries, seeded 20260929,
cudnn.benchmark=False. Stages: preprocessing/H2D, forward inference, postprocessing
(NMS/native masks/bit packing/CPU transfer/prediction objects); pipeline is their sum.
RLE preparation separate. Exclude load, disk/decode, GT, evaluator, serialization and visualization.
Peak allocated VRAM includes resident model, reset after warmup; maximum across clean rounds.
FPS=1000/mean pipeline ms, pooled 300 observations; population std, P50/P95.
GPU monitored before/during/after; other compute PID or idle utilization >5% contaminates
whole round, preserve/exclude and repeat only missing rounds in a fresh attempt directory.
All framework paths stay inside this experiment. No mid-run reports/plots.

## Documentation scope

The user's Small request explicitly selects the supplied numerical RESULTS and numbered
PRESENTATION headings, superseding the newer qualitative template layout for this tier only.
Snapshot templates under configs/report_templates/; keep canonical measured values unchanged.
No mentor-specific text, weighted score, statistical-superiority or final CCTV claims.
Update study state only after scientific/document validation. STOP after Small; never launch Nano.
