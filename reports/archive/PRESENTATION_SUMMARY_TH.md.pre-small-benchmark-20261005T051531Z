# Small (S) — Visual and Qualitative Analysis

<!-- Purpose: เมื่อดู prediction จริง โมเดลต่างกันอย่างไร?
Only analyze after all tier models complete and artifacts are validated.
For NOT_RUN/incomplete tiers, explicitly mark pending, retain empty tables, and
do not embed placeholder image links or analyze one completed model alone. -->

## 1. ภาพรวมผลการทดลอง

ยังไม่รัน (NOT_RUN) — รอผลครบทุกโมเดลและการตรวจ canonical artifacts ก่อนสรุป

| Model | Mask mAP50-95 | AP75 | Recall |
|---|---|---|---|
<!-- Context only. Full quantitative summary: RESULTS_SUMMARY_TH.md. -->

## การเลือกกรณีและการอ่านภาพ

ยังไม่รัน (NOT_RUN) — รอผลครบทุกโมเดลและการตรวจ canonical artifacts ก่อนสรุป
<!-- Select ~3–5 diagnostic cases from actual benchmark artifacts: advantage,
shared failure, counterexample/trade-off, similar-output/near-tie where available.
Shortlist with existing per-frame TP/FP/FN/matched IoU; inspect few candidates.
First reuse comparisons; otherwise render saved predictions + original/GT.
Never rerun inference for documentation. If reconstruction unavailable, report it.
Same frame/full region, scale, confidence and panel order for all tier models:
Original / GT, YOLO26, YOLO11, valid YOLOv9 e/c for X/E and L/C only, YOLOv8.
Readable labels; explain ignore handling and FN/FP semantics. Retain source hashes.
CASE_SELECTION.md records sequence/frame/reason and limitations of sampling. -->

## กรณีภาพจริง

ยังไม่รัน (NOT_RUN) — รอผลครบทุกโมเดลและการตรวจ canonical artifacts ก่อนสรุป

## Failure Analysis

| Failure pattern | Models observed | Visual case | Interpretation |
|---|---|---|---|
<!-- Observed categories only; no unsupported frequency/counts. FP means unmatched
under benchmark policy, not necessarily a nonexistent person. -->

## Near-tie visual check

ยังไม่รัน (NOT_RUN) — รอผลครบทุกโมเดลและการตรวจ canonical artifacts ก่อนสรุป
<!-- No automatic cutoff or statistical superiority. If only timing is close,
state that segmentation images cannot verify a latency near tie. -->

## สิ่งที่เรียนรู้จากภาพจริง

ยังไม่รัน (NOT_RUN) — รอผลครบทุกโมเดลและการตรวจ canonical artifacts ก่อนสรุป

## เมื่อดูทั้งตัวเลขและภาพร่วมกัน

ยังไม่รัน (NOT_RUN) — รอผลครบทุกโมเดลและการตรวจ canonical artifacts ก่อนสรุป
Latency และ VRAM เป็น system-level measurements อ่านจาก benchmark;
ภาพ segmentation ไม่สามารถอธิบายหรือวัดสองค่านี้ได้

## ถ้าพิจารณาทั้งผลเชิงตัวเลขและภาพ

| Priority | Candidate | Evidence |
|---|---|---|
<!-- Accuracy: quantitative + visual, Speed: clean benchmark latency,
Low VRAM: memory benchmark, Balanced: explicit trade-off/constraints.
No weighted score and no final CCTV superiority. -->

## ข้อจำกัด

- เมื่อมีผลครบ เฟรมที่เลือกจะเป็นตัวอย่างเชิงคุณภาพ ไม่แทน dataset-level metrics
- ต้องเลือกทั้งข้อได้เปรียบ ข้อผิดพลาด กรณีสวนอันดับ และผลคล้ายกัน เพื่อลด cherry-picking; ขณะนี้ยังไม่มีกรณีที่เลือก
- MOTS20 ไม่ใช่ผลทดสอบ CCTV robustness ขั้นสุดท้าย
- Qualitative observations และ numerical near ties ไม่ใช่ statistical significance

## รายละเอียดเต็ม

[Quantitative summary](RESULTS_SUMMARY_TH.md) · [REPORT.md](REPORT.md) ·
[TIER_RESULTS.csv](metrics/TIER_RESULTS.csv) ·
[Master Study](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study)
