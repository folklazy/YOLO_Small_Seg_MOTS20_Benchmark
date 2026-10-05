# Small qualitative case selection

Pool: the 12 frames in `manifests/visualization_frames.json`, frozen before full inference. Shortlisted using existing per-frame TP/FP/FN and matched-mask IoU; inspected only the four selected comparisons. This is diagnostic selection, not representative sampling. No model inference, confidence changes, crops or measured-value changes.

| Case | Sequence / frame | Reason and behavior |
|---|---|---|
| 1 | MOTS20-02 / 000300 | YOLO26s 10 TP versus 7/8; shared misses and distinct small-person errors in YOLO11s/YOLOv8s. |
| 2 | MOTS20-09 / 000263 | Shared FN 2002/2011/2023 and unmatched predictions on people; all models still fail. |
| 3 | MOTS20-02 / 000600 | Counterexample: YOLO11s/YOLOv8s 10 TP, 1 FP, 0 FN versus YOLO26s 9 TP, 3 FP, 1 FN; similar near-tie pair. |
| 4 | MOTS20-09 / 000001 | All recover 6 valid GT; YOLO11s has one additional background FP despite similar Person masks. |

Original/GT followed by YOLO26s, YOLO11s, YOLOv8s, full frame at identical scale. Confidence 0.25, mask IoU 0.50, ignore prediction IOA 0.50. FP means unmatched under the benchmark, not necessarily a nonexistent person; FN can mean a mask did not pass matching. TP-only means need not compare identical matched GT sets.

`CASE_EVIDENCE.json` records original/GT/prediction/comparison hashes and verified per-frame counts. Source: `metrics/benchmark-20261005T051531Z/per_frame/` and saved lossless RLE predictions. Renderer: Master Study `src/build_qualitative_comparisons.py --tier Small`. No observed merging/fragmentation/boundary leakage category is asserted without evidence.
