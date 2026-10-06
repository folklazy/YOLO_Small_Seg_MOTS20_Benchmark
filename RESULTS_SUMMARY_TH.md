# สรุปผล Small YOLO Instance Segmentation

## สรุปใน 1 นาที

- ทดสอบ YOLO26s-Seg, YOLO11s-Seg และ YOLOv8s-Seg สำหรับ Person instance segmentation
- MOTS20 2,862 frames / 26,894 Person GT instances รายเฟรม; pretrained / no fine-tuning
- Accuracy สูงสุด: YOLO26s-Seg — mAP50-95 0.536640; นำ AP75 และ Recall ด้วย
- Inference เร็วสุด: YOLOv8s-Seg — 15.220 ms
- Pipeline เร็วสุด/FPS สูงสุด: YOLO26s-Seg — 78.828 ms / 12.686 FPS
- Peak allocated VRAM ต่ำสุด: YOLO26s-Seg — 872.67 MiB
- Trade-off หลัก: YOLO26s นำ YOLO11s ด้าน mAP 0.053371 แต่ forward ช้ากว่า YOLOv8s 2.088 ms; YOLO11s/YOLOv8s มี mAP near-tied เชิงพรรณนา

## ผลลัพธ์หลัก

| Model | Mask mAP50-95 | AP75 | Recall | Inference ms | Pipeline ms | FPS | Peak VRAM MiB |
|---|---|---|---|---|---|---|---|
| YOLO26s-Seg | 0.536640 | 0.586796 | 0.789581 | 17.308 | 78.828 | 12.686 | 872.67 |
| YOLO11s-Seg | 0.483269 | 0.513823 | 0.764148 | 15.693 | 88.836 | 11.257 | 1007.50 |
| YOLOv8s-Seg | 0.482763 | 0.506411 | 0.767309 | 15.220 | 92.173 | 10.849 | 1139.01 |

## สรุปผลจากตาราง

AP50 และ TP-only IoU/Dice ที่ไม่อยู่ในตารางย่ออ้างอิง [canonical CSV](metrics/TIER_RESULTS.csv) และ [REPORT.md](REPORT.md) TP-only quality วัดเฉพาะคู่ที่ match และอาจใช้ GT คนละชุดระหว่างโมเดล จึงต้องอ่านร่วมกับ Recall

### YOLO26s-Seg

นำ AP50, AP75, mAP50-95, Recall และ TP-only IoU/Dice พร้อมกัน จึงมีหลักฐานหลายด้านทั้งความครอบคลุม GT และ overlap ของคู่ที่ match โดย Recall 78.96% ยังเหลือ GT ที่พลาดหรือจับคู่ไม่ผ่าน ไม่ใช่การเก็บ Person ครบทุกคน

สิ่งที่แลกคือ inference 17.308 ms ช้าที่สุด แต่ pipeline 78.828 ms เร็วที่สุดและ peak allocated VRAM 872.67 MiB ต่ำที่สุด จึงเป็น candidate เมื่อเน้น accuracy ร่วมกับ pipeline และ memory; ถ้างานจำกัดเวลา forward โดยตรง ต้องเทียบ YOLOv8s เพิ่ม

### YOLO11s-Seg

mAP 0.483269 แทบเท่า YOLOv8s 0.482763 แต่ AP75 และ TP-only IoU/Dice สูงกว่าเล็กน้อย ขณะที่ Recall ต่ำกว่า จึงไม่ควรใช้ mAP ทศนิยมท้าย ๆ เป็นหลักฐานความเหนือกว่า หรือมองว่าผลทั้งสองเหมือนกันทุกด้าน

Pipeline 88.836 ms และ allocated VRAM 1007.50 MiB ดีกว่า YOLOv8s แต่ inference 15.693 ms ช้ากว่า เหมาะเป็น candidate เมื่อพิจารณาคู่นี้ภายใต้ข้อจำกัด pipeline/memory; เมื่อเทียบ YOLO26s ยังมี mAP ต่ำกว่า พร้อม pipeline ช้ากว่าและ VRAM สูงกว่า จึงไม่เรียก “สมดุลดีที่สุด” โดยอัตโนมัติ

### YOLOv8s-Seg

Inference 15.220 ms เร็วที่สุด และ Recall 0.767309 สูงกว่า YOLO11s แต่ mAP/AP75 และ TP-only quality ต่ำสุดในกลุ่ม โดย mAP ใกล้ YOLO11s มาก ต้องแยกความต่างเล็กนี้ออกจากข้อสรุปทางสถิติ

Pipeline 92.173 ms ช้าที่สุดและ allocated VRAM 1139.01 MiB สูงที่สุด จึงเป็น candidate เมื่อ forward latency เป็นข้อจำกัดหลัก มากกว่าจะใช้ forward แทน throughput ทั้ง pipeline ผลนี้จำกัดเฉพาะ checkpoint และ protocol รอบนี้ ไม่ยืนยันความเหนือกว่าของ architecture หรือความพร้อมใช้งาน CCTV

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

- YOLO26s นำ YOLO11s ด้าน mAP 0.053371 บนสเกล 0–1 พร้อม AP75/Recall/TP-only quality สูงสุด
- YOLO11s/YOLOv8s มี mAP ต่างเพียง 0.000505; YOLO11s นำ AP75/TP-only quality แต่ YOLOv8s นำ Recall เป็น near tie เชิงพรรณนา ไม่มี significance test
- อันดับ forward กับ pipeline ต่างกัน: YOLOv8s forward เร็วสุด แต่ pipeline ช้ากว่า YOLO26s 13.345 ms
- YOLO26s ใช้ allocated VRAM ต่ำกว่า YOLO11s/YOLOv8s 134.82/266.34 MiB; YOLO11s มี parameters ต่ำสุดตาม [MODEL_COMPLEXITY.csv](metrics/MODEL_COMPLEXITY.csv) แต่ไม่ได้ใช้ VRAM ต่ำสุด

## Trade-off หลัก

### Accuracy vs Speed

YOLO26s มี mAP สูงสุดและ forward ช้ากว่า YOLOv8s 2.088 ms แต่ pipeline เร็วกว่า 13.345 ms จึงต้องเลือก stage ให้ตรงกับงาน ค่า mean postprocessing ของ YOLO26s/YOLO11s/YOLOv8s คือ 59.810/71.443/75.255 ms ตาม [TIMING_SUMMARY.csv](metrics/TIMING_SUMMARY.csv) เป็นการแจกแจงเวลาที่วัดได้ ไม่ใช่หลักฐานเหตุเชิง architecture

### Accuracy vs Memory

YOLO26s มี accuracy สูงกว่าและใช้ peak allocated VRAM น้อยกว่าอีกสองโมเดลใน protocol นี้ จึงไม่มีการแลก accuracy กับ VRAM เพิ่มเมื่อเลือกตัวนำ ส่วนคู่ near-tied YOLO11s/YOLOv8s ต้องเลือกระหว่าง pipeline/VRAM ที่ดีกว่าของ YOLO11s กับ Recall/forward ที่ดีกว่าของ YOLOv8s

## ข้อควรระวังในการตีความ

PASS WITH WARNINGS จาก CPU NNPACK ใน log; ผลเป็น Person instance segmentation รายเฟรม ไม่ใช่ tracking หรือจำนวนคนไม่ซ้ำ TP-only quality มีเงื่อนไขการ match; AP/Recall ใช้สเกล 0–1 และไม่มี statistical-significance test Pipeline FPS ไม่รวม RLE preparation และ disk I/O; VRAM คือ peak allocated ของ benchmark MOTS20 ยังไม่ใช่ผล CCTV robustness ขั้นสุดท้าย

## ข้อมูลสำหรับนำไปรวมต่อ

นำ YOLO26s สำหรับ accuracy/pipeline/VRAM และ YOLOv8s สำหรับ forward ไปพิจารณาข้าม tier; เก็บ YOLO11s/YOLOv8s เป็นคู่ near-tie ที่มี trade-off ต่างกัน ไม่ใช้ weighted score หรือสรุป final CCTV superiority

[TIER_RESULTS.csv](metrics/TIER_RESULTS.csv) · [REPORT.md](REPORT.md) · [Visual analysis](PRESENTATION_SUMMARY_TH.md) · [Master Study](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study)
