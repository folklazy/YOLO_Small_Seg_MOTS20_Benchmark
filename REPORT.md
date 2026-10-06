# Small การทดสอบ YOLO Instance Segmentation — MOTS20

## 1. สถานะการทดลอง

PASS WITH WARNINGS

- โมเดลที่เสร็จแล้ว: 3/3
- จำนวนเฟรม: 2,862 ต่อโมเดล; Person GT รายเฟรม: 26,894 instances
- รหัสรอบทดลอง: `benchmark-20261005T051531Z`

## 2. โมเดลที่ทดสอบ

| ตระกูล | โมเดล | จำนวนพารามิเตอร์ | GFLOPs | Checkpoint (MB) |
| --- | --- | --- | --- | --- |
| YOLO26 | YOLO26s-Seg | 11,505,800 | 37.662 | 23.47 |
| YOLO11 | YOLO11s-Seg | 10,113,248 | 33.364 | 20.67 |
| YOLOv8 | YOLOv8s-Seg | 11,821,056 | 40.339 | 23.91 |

## 3. ความสอดคล้องกับโพรโทคอล

| รายการ | สถานะ |
|---|---|
| ข้อมูล | PASS |
| ตัวประเมิน | PASS |
| การเตรียมภาพ | PASS |
| ขนาดภาพเข้าโมเดล | PASS |
| ความละเอียดเชิงตัวเลข | PASS |
| ค่าเกณฑ์ | PASS |
| maxDet | PASS |
| วิธีวัดเวลา | PASS |
| สภาพแวดล้อม | PASS |

ความสอดคล้องของข้อมูล: PASS

ความสอดคล้องของการเตรียมภาพ: PASS

[วิธีทดลองร่วม](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study/blob/main/METHODOLOGY_REFERENCE.md) · [โพรโทคอลการทดลอง](EXPERIMENT_PROTOCOL.md) · [หลักฐานการกำหนดมาตรฐาน](manifests/STANDARDIZATION.json)

ผลตรวจความถูกต้องทางวิทยาศาสตร์: PASS; รายการตรวจละเอียดทั้ง 15 ข้อยังคงอยู่ใน [หลักฐานตรวจรอบทดลอง](manifests/final_integrity.json)

## 4. ผลลัพธ์รวม

| โมเดล | Mask mAP50-95 | AP50 | AP75 | Precision | Recall | F1 | TP-only IoU | TP-only Dice | Inference (ms) | Pipeline (ms) | FPS | Peak allocated VRAM (MiB) | จำนวนพารามิเตอร์ | GFLOPs | Checkpoint (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s-Seg | 0.536640 | 0.837980 | 0.586796 | 0.890842 | 0.789581 | 0.837161 | 0.816347 | 0.894970 | 17.308 | 78.828 | 12.686 | 872.67 | 11,505,800 | 37.662 | 23.47 |
| YOLO11s-Seg | 0.483269 | 0.817429 | 0.513823 | 0.868267 | 0.764148 | 0.812887 | 0.793011 | 0.880512 | 15.693 | 88.836 | 11.257 | 1007.50 | 10,113,248 | 33.364 | 20.67 |
| YOLOv8s-Seg | 0.482763 | 0.817024 | 0.506411 | 0.867022 | 0.767309 | 0.814124 | 0.792418 | 0.880020 | 15.220 | 92.173 | 10.849 | 1139.01 | 11,821,056 | 40.339 | 23.91 |

## 5. ผู้ชนะในแต่ละด้าน

| ด้าน | โมเดล | ผลลัพธ์ |
| --- | --- | --- |
| Mask mAP50-95 สูงสุด | YOLO26s-Seg | 0.536640 |
| AP75 สูงสุด | YOLO26s-Seg | 0.586796 |
| Recall สูงสุด | YOLO26s-Seg | 0.789581 |
| Inference เร็วสุด | YOLOv8s-Seg | 15.220 |
| Pipeline เร็วสุด | YOLO26s-Seg | 78.828 |
| FPS สูงสุด | YOLO26s-Seg | 12.686 |
| VRAM ต่ำสุด | YOLO26s-Seg | 872.67 |

## 6. ข้อค้นพบสำคัญ

- ข้อสังเกต: YOLO26s-Seg มี Mask mAP50-95 สูงสุด 0.536640; ห่างอันดับถัดไป 0.053371 บนสเกล 0–1
- ข้อสังเกต: YOLO26s-Seg นำ AP75; YOLO26s-Seg นำ Recall
- ข้อสังเกต: YOLOv8s-Seg มี inference เร็วสุด; YOLO26s-Seg มี pipeline เร็วสุดและ FPS สูงสุด; YOLO26s-Seg มี VRAM ต่ำสุด
- คู่ mAP ใกล้ที่สุด: YOLO11s-Seg / YOLOv8s-Seg ต่าง 0.000505; เป็นความใกล้เชิงพรรณนา ไม่ใช่ผลทดสอบนัยสำคัญทางสถิติ
- การตีความ: แยกความแม่นยำความครบถ้วนเวลา forward เวลา pipeline และหน่วยความจำไม่มีคะแนนรวมถ่วงน้ำหนักจำนวนพารามิเตอร์หรือ GFLOPs ไม่กำหนดอันดับเวลา/VRAM โดยตรง

## 7. ข้อสังเกตรายลำดับภาพ

- YOLO26s-Seg: Mask mAP50-95 สูงสุดที่ MOTS20-11 (0.589893); ต่ำสุดที่ MOTS20-02 (0.419651)
- YOLO11s-Seg: Mask mAP50-95 สูงสุดที่ MOTS20-05 (0.539178); ต่ำสุดที่ MOTS20-02 (0.356259)
- YOLOv8s-Seg: Mask mAP50-95 สูงสุดที่ MOTS20-11 (0.541606); ต่ำสุดที่ MOTS20-02 (0.359863)
- ลำดับ mAP ที่ต่างจากผลรวม: MOTS20-02: YOLO26s-Seg > YOLOv8s-Seg > YOLO11s-Seg; MOTS20-11: YOLO26s-Seg > YOLOv8s-Seg > YOLO11s-Seg

AP รวมคำนวณจากข้อมูลทั้งหมด ไม่ใช่ค่าเฉลี่ย AP รายลำดับภาพ ดูค่าครบใน [PER_SEQUENCE_RESULTS.csv](metrics/PER_SEQUENCE_RESULTS.csv)

## 8. ประสิทธิภาพและการใช้ทรัพยากร

- คู่ที่ใกล้ที่สุดด้านค่าเฉลี่ย inference: YOLO11s-Seg / YOLOv8s-Seg ต่าง 0.473 ms; ไม่ได้ทดสอบนัยสำคัญทางสถิติ
- คู่ที่ใกล้ที่สุดด้านค่าเฉลี่ย pipeline: YOLO11s-Seg / YOLOv8s-Seg ต่าง 3.337 ms; ไม่ได้ทดสอบนัยสำคัญทางสถิติ

YOLO26s-Seg ใช้ peak allocated VRAM ต่ำสุดจำนวนพารามิเตอร์ก่อน/หลัง fusion, GFLOPs และเวลาโหลดแยกเก็บใน [MODEL_COMPLEXITY.csv](metrics/MODEL_COMPLEXITY.csv) ส่วน peak reserved VRAM อยู่ใน [แหล่งวัดเวลา](timing/benchmark-20261005T051531Z/clean_repetition/summary.csv) เวลาเตรียม RLE แยก: yolo26s-seg.pt: 279.956 ms; yolo11s-seg.pt: 330.744 ms; yolov8s-seg.pt: 344.705 ms.

ใช้ 3 รอบที่ไม่ถูกรบกวนต่อโมเดล รอบละ 100 เฟรมหลัง 10 warmups และ synchronize CUDA ตามขอบเขต stage ค่า pipeline รวม preprocessing, inference และ postprocessing ไม่รวมการเตรียม RLE และการอ่านเขียนดิสก์ FPS จึงไม่ใช่อัตราการบันทึก mask ครบกระบวนการและไม่บวก Ultralytics-inclusive diagnostic ซ้ำ

ค่าเฉลี่ย postprocessing: YOLO26s-Seg: 59.810 ms; YOLO11s-Seg: 71.443 ms; YOLOv8s-Seg: 75.255 ms

## 9. คำเตือนและข้อสังเกตผิดปกติ

พบคำเตือน CPU NNPACK ในรอบ Small ไม่พบ pycocotools DeprecationWarning ใน log ของ Small จึงไม่ถือคำเตือนจากขนาดก่อนหน้าเป็นคำเตือนใหม่ การตรวจ regression ของตัวประเมินผ่าน ไม่เปลี่ยน package เพื่อซ่อนคำเตือน ผลเวลาหลักมี 9 รอบที่ไม่ถูกรบกวน

Pipeline ไม่รวมการเตรียม RLE และการอ่านเขียนดิสก์จึงไม่ใช่เวลา/อัตราประมวลผลครบกระบวนการสำหรับการบันทึก mask หรือระบบ CCTV การปรับเอกสารครั้งนี้ไม่รัน inference ใหม่และไม่เปลี่ยนค่าที่วัด

## 10. ข้อจำกัด

ผลนี้เป็น Person instance segmentation รายเฟรมบน MOTS20 ไม่ใช่ MOTS tracking; 26,894 GT instances เป็น annotation รายเฟรมไม่ใช่จำนวนคนไม่ซ้ำ TP-only IoU/Dice พิจารณาเฉพาะคู่ที่ จับคู่ ได้ ภาพต่อเนื่องสัมพันธ์กันและไม่มีการทดสอบนัยสำคัญทางสถิติผลยังไม่ยืนยันภาพพร่า, แสงน้อย, มุมกล้อง, ระดับ occlusion หรือความเหมาะสมต่อการนำไปใช้งานจึงใช้เพื่อเลือกตัวเลือกสำหรับการทดสอบต่อการประเมินความทนทานต่อ CCTV เท่านั้น

## 11. หลักฐานสำหรับตรวจสอบซ้ำ

- [TIER_RESULTS.csv](metrics/TIER_RESULTS.csv)
- [PER_SEQUENCE_RESULTS.csv](metrics/PER_SEQUENCE_RESULTS.csv)
- [TIMING_SUMMARY.csv](metrics/TIMING_SUMMARY.csv)
- [MODEL_COMPLEXITY.csv](metrics/MODEL_COMPLEXITY.csv)
- [PREFLIGHT_MAXDET.csv](metrics/PREFLIGHT_MAXDET.csv)

[แหล่งที่มาและค่า hash](manifests/STANDARDIZATION.json) · [โพรโทคอล](EXPERIMENT_PROTOCOL.md) · [รายการกราฟ](outputs/plots/INDEX.md) · [บันทึกย้อนหลัง](reports/archive/)

prediction แบบ RLE ที่ไม่สูญเสียข้อมูลและบันทึกการวัดเวลาละเอียดเก็บในเครื่องตามรหัสรอบทดลอง หลักฐานต้นทางคงเดิม; การปรับภาษานี้ไม่คำนวณค่าตัวชี้วัดใหม่และไม่รัน inference

## 12. ความเชื่อมโยงกับการศึกษาทุกขนาด

[การศึกษาหลัก](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study) — รายงานนี้กล่าวถึงขนาด Small (S) เท่านั้นผลรวม 17 โมเดลยังรอคำสั่งจากผู้ใช้ แม้การทดลองทั้งห้าขนาดเสร็จแล้ว การปรับเอกสารไม่เริ่ม benchmark หรือการสังเคราะห์ผลใหม่

## การวิเคราะห์เชิงคุณภาพ

ภาพเปรียบเทียบเฟรมเดียวกัน 4 กรณีจาก prediction ที่บันทึกไว้ พร้อมข้อผิดพลาดที่พบและการตีความ อยู่ใน [PRESENTATION_SUMMARY_TH.md](PRESENTATION_SUMMARY_TH.md) ดู [เหตุผลเลือกกรณีปัจจุบัน](outputs/visualizations/qualitative/selection_v2/CASE_SELECTION.md) และ [ตัวชี้ชุดหลักฐาน](manifests/QUALITATIVE_SELECTION.json) รายงานเทคนิคนี้เชื่อมไปยังการวิเคราะห์ภาพเพื่อไม่เล่าเนื้อหาซ้ำ
