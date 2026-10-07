# สรุปผล Small YOLO Instance Segmentation

## สรุปใน 1 นาที

- โมเดล: YOLO26s-Seg, YOLO11s-Seg, YOLOv8s-Seg
- MOTS20 2,862 frames / 26,894 Person GT instances รายเฟรม
- Official pretrained checkpoints; ไม่มี training หรือ fine-tuning; สถานะ PASS WITH WARNINGS
- Accuracy สูงสุด: YOLO26s-Seg — Mask mAP50-95 0.536640
- Inference เร็วสุด: YOLOv8s-Seg — 15.220 ms
- Pipeline เร็วสุด: YOLO26s-Seg — 78.828 ms / 12.686 FPS
- Peak allocated VRAM ต่ำสุด: YOLO26s-Seg — 872.67 MiB
- Trade-off หลัก: ตัวนำ mAP สูงกว่ารองอันดับสอง 5.337 percentage points; ต้องแยก forward จาก pipeline

## ผลลัพธ์หลัก

| Model | Mask mAP50-95 | AP75 | Recall | Inference ms | Pipeline ms | FPS | Peak VRAM MiB |
|---|---|---|---|---|---|---|---|
| YOLO26s-Seg | 0.536640 | 0.586796 | 0.789581 | 17.308 | 78.828 | 12.686 | 872.67 |
| YOLO11s-Seg | 0.483269 | 0.513823 | 0.764148 | 15.693 | 88.836 | 11.257 | 1007.50 |
| YOLOv8s-Seg | 0.482763 | 0.506411 | 0.767309 | 15.220 | 92.173 | 10.849 | 1139.01 |

AP/Recall เป็น fraction ช่วง 0–1; latency เป็น ms/frame และ FPS มาจาก mean pipeline

## Winner ของแต่ละด้าน

| ด้าน | Model | Result |
|---|---|---|
| Mask mAP50-95 | YOLO26s-Seg | 0.536640 |
| AP75 | YOLO26s-Seg | 0.586796 |
| Recall | YOLO26s-Seg | 0.789581 |
| Inference speed | YOLOv8s-Seg | 15.220 ms |
| Pipeline speed | YOLO26s-Seg | 78.828 ms |
| VRAM | YOLO26s-Seg | 872.67 MiB |

## สิ่งที่ตัวเลขบอกเรา

- YOLO26s-Seg นำ YOLO11s-Seg ด้าน mAP 5.337 percentage points
- YOLO11s/YOLOv8s เป็น descriptive near tie ของ mAP แต่ AP75, Recall และ resource ranking ต่างกัน
- YOLO26s มี forward ช้าที่สุดใน tier แต่ pipeline เร็วที่สุด; postprocessing เป็นส่วนหนึ่งของผลรวม
- YOLO11s-Seg/YOLOv8s-Seg: ต่าง 0.050536 percentage points; near tie ไม่ใช่ equivalence หรือ statistical significance

## บทบาทของแต่ละโมเดล

| Model | จุดเด่น | สิ่งที่แลก | เหมาะพิจารณาเมื่อ |
|---|---|---|---|
| YOLO26s-Seg | นำ mAP/AP75/Recall; pipeline/VRAM ต่ำสุด | Forward ช้าที่สุดใน tier | เน้น accuracy ของ Small และ native-mask pipeline |
| YOLO11s-Seg | mAP near-tied กับ YOLOv8s; pipeline/VRAM ต่ำกว่า | Recall และ forward ด้อยกว่า YOLOv8s | ตรวจ resource trade-off ภายในคู่ near tie |
| YOLOv8s-Seg | Forward เร็วสุด; Recall สูงกว่า YOLO11s | Pipeline/VRAM สูงสุด; AP75 ต่ำกว่า YOLO11s | สนใจ forward/coverage พร้อมยอมรับต้นทุน pipeline |

## Trade-off หลัก

### Accuracy vs Speed

YOLO26s-Seg มี mAP 0.536640; ตัว forward เร็วสุด YOLOv8s-Seg มี mAP 0.482763 และ inference ต่ำกว่า 2.088 ms ส่วน pipeline ต้องดู YOLO26s-Seg แยก ไม่ถือว่า forward winner เป็น throughput winner

### Accuracy vs Memory

YOLO26s-Seg นำทั้ง mAP และ allocated VRAM ต่ำสุดใน tier นี้ จึงไม่มีการแลก accuracy ลงเพื่อ memory ที่ต่ำกว่าในคู่ที่วัด ไม่ใช้ชื่อขนาดหรือ parameters แทน memory measurement

## ข้อควรระวังในการตีความ

ไม่มี significance test; ภาพวิดีโอสัมพันธ์กัน TP-only quality วัดเฉพาะคู่ที่ match และ Recall เป็น mask matching ไม่ใช่ box Recall Pipeline ไม่รวม decode, RLE preparation และการเขียนผล; VRAM เป็น peak allocated ภายใต้ benchmark นี้ การแบ่ง tier ไม่ทำให้ capacity/pretraining เท่ากัน และยังไม่ยืนยัน CCTV robustness ไม่มี weighted score หรือผู้ชนะทุกข้อจำกัด

## รายละเอียดเพิ่มเติม

[REPORT.md](REPORT.md) · [รายงานวิจัยภาพเชิงคุณภาพ](PRESENTATION_SUMMARY_TH.md) · [TIER_RESULTS.csv](metrics/TIER_RESULTS.csv) · [Master Study](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study)
