# Small — qualitative case selection v2

Selected from 12 frozen visualization frames using per-frame metrics, original/GT images and saved RLE masks at confidence ≥0.25 and mask matching IoU ≥0.50, with the original evaluator and ignore policy. No inference was run.

One shared anchor plus three cases selected for tier-specific behavior. Tiers need not share every frame; within each case, all models use the same full frame. ROI views supplement the full comparison and do not hide errors elsewhere.

| Case | Sequence / frame | Why selected / decision use | Source comparison |
|---|---|---|---|
| 1 | MOTS20-02 / 000300 | Tier diagnostic: largest TP spread in the frozen candidate pool. YOLO26s/YOLO11s/YOLOv8s have TP 10/7/8 and FP 1/3/3, with different matched GT locations; use this to compare coverage. | reuse existing image |
| 2 | MOTS20-09 / 000263 | Shared anchor: common failure. There are common FN, and the near-tied mAP pair may miss different people; assess coverage and unmatched outputs together. | reuse existing image |
| 3 | MOTS20-02 / 000600 | Counterexample and near-tied pair. YOLO11s and YOLOv8s recover all 10 valid GT in this frame, while YOLO26s misses GT 2043 and produces an extra FP. | reuse existing image |
| 4 | MOTS20-11 / 000900 | Near-tied pair: different missed GT and extra outputs. YOLO11s matches GT 2065 with no FP; YOLOv8s misses GT 2065 and has two FP masks. YOLO26s additionally recovers GT 2064 but still has an FP. | new composite from saved predictions |

## Why some frames are shared across tiers

Case 2 (09/263) compares FN/FP against the same GT. Other frames may repeat when the same error region helps compare different models: 05/419 examines GT 2002 in Second-largest/Medium; 02/1 examines equal counts and different GT sets in Second-largest/Medium; 02/600 provides a counterexample in Largest/Small; 02/300 examines TP–FP trade-offs in Small/Nano; 11/1 examines GT 2016 in Second-largest and GT 2028 with extra masks in Nano; 11/450 compares equal coverage with extra outputs in Largest/Medium. Reused frames are not additional independent evidence.

## Replacement of previous cases

Previous Case 4 (09/1) showed a useful background FP. Its replacement (11/900) adds a near-tie check with differences in both FN and FP. Case 3 retains similar outputs and a counterexample to the aggregate ranking. Previous images and evidence remain unchanged for audit; the previous presentation is retained under reports/archive.

## Scope

Across five tiers, 20 case slots use 10 distinct original frames (previously 6). The selection includes MOTS20-11, common failures and counterexamples, rather than only frames where the accuracy leader wins. The 12-frame pool does not represent the dataset, and these are not claimed to be the most divergent frames among all 2,862. Images do not measure latency, VRAM or statistical significance.

[Candidate pool](CANDIDATE_POOL.json) · [Case evidence](CASE_EVIDENCE.json) · [Focus evidence](FOCUS_EVIDENCE.json) · [Decision audit](CASE_DECISION_AUDIT.json) · [Active selection](../../../../manifests/QUALITATIVE_SELECTION.json)
