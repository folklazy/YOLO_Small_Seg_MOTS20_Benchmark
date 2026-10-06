# Small — qualitative case selection v2

คัดจาก 12 frozen visualization frames โดยตรวจ per-frame metrics, original/GT และ saved RLE ที่ confidence ≥0.25 / mask matching IoU ≥0.50 ตาม evaluator/ignore policy เดิม ไม่รัน inference

1 shared anchor + 3 cases ตามพฤติกรรมของ tier ไม่บังคับภาพทั้งหมดตรงกันระหว่าง tier; ภายใน case ใช้เฟรมเต็มเดียวกันทุกโมเดล ROI เป็นภาพเสริม ไม่ซ่อน full-frame errors

| Case | Sequence / frame | Why selected / decision use | Source comparison |
|---|---|---|---|
| 1 | MOTS20-02 / 000300 | tier diagnostic: largest TP spread in frozen pool; เป็น case ที่ช่วยเลือกด้าน coverage ได้ชัด: TP 10/7/8 และ FP 1/3/3 ของ YOLO26s/YOLO11s/YOLOv8s พร้อมตำแหน่ง GT ที่ต่างกัน | reuse existing image |
| 2 | MOTS20-09 / 000263 | shared anchor: common failure; ช่วยเห็นว่ายังมี FN ร่วม และ pair ที่ mAP near-tied อาจพลาดคนคนละชุด; ใช้ตรวจความครบถ้วนและ unmatched output ร่วมกัน | reuse existing image |
| 3 | MOTS20-02 / 000600 | counterexample and near-tied pair; ช่วยตรวจ pair near-tied YOLO11s/YOLOv8s ซึ่งเก็บ valid GT ครบ 10 ในเฟรมนี้ แต่ YOLO26s มี FN 2043 และ FP เพิ่ม | reuse existing image |
| 4 | MOTS20-11 / 000900 | near-tied pair: different missed GT and extra outputs; ใช้เทียบ pair near-tied บนคนเดียวกัน: YOLO11s match 2065 ได้และไม่มี FP แต่ YOLOv8s พลาด 2065 พร้อม FP สอง mask; YOLO26s เก็บ 2064 เพิ่มแต่ยังมี FP | new composite from saved predictions |

## ทำไมบางภาพยังตรงกับ tier อื่น

Case 2 (09/263) ใช้ร่วมเพื่อเทียบ FN/FP บน GT ชุดเดียวกัน กรณีอื่นซ้ำได้เมื่อ error เดียวกันช่วยตรวจคนละโมเดล: 05/419 ใช้ L/M ตรวจ GT 2002; 02/1 ใช้ L/M ตรวจ equal counts และ GT ต่างชุด; 02/600 ใช้ Largest/Small ตรวจกรณีสวนอันดับ; 02/300 ใช้ Small/Nano ตรวจ TP–FP trade-off; 11/1 ใช้ L/N แต่ L ตรวจ GT 2016 ส่วน N ตรวจ GT 2028 และ extra mask; 11/450 ใช้ Largest/Medium ตรวจ coverage เท่ากันกับ extra output ของคนละชุดโมเดล ไม่ใช้จำนวนภาพซ้ำเป็นหลักฐานอิสระเพิ่ม

## การแทน case เดิม

เดิม Case 4 (09/1) มี FP ฉากหลังที่น่าสนใจ แต่ 11/900 เพิ่ม near-tie check ที่แสดงทั้ง FN/FP ต่างกัน; Case 3 ยังแสดงผลคล้ายกันและกรณีสวนอันดับ ภาพ/หลักฐานเก่ายังคงเดิมเพื่อ audit; presentation เก่าเก็บใน reports/archive

## ขอบเขต

ทั้งห้า tier มี 20 case slots แต่ใช้ original frames ต่างกัน 10 เฟรม (เดิม 6) ชุดใหม่มี MOTS20-11 และยังมี common failure / counterexample ไม่เลือกเฉพาะ frame ที่ accuracy leader ชนะ ทั้งนี้ pool 12 เฟรมไม่แทน dataset; ไม่อ้างว่าเป็นเฟรมที่ต่างที่สุดใน 2,862 เฟรม ไม่ใช้ภาพวัด latency/VRAM หรือ statistical significance

[Candidate pool](CANDIDATE_POOL.json) · [Case evidence](CASE_EVIDENCE.json) · [Focus evidence](FOCUS_EVIDENCE.json) · [Decision audit](CASE_DECISION_AUDIT.json) · [Active selection](../../../../manifests/QUALITATIVE_SELECTION.json)
