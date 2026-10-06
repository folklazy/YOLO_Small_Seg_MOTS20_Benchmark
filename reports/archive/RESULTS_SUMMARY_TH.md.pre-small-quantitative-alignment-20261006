# สรุปผล Small YOLO Instance Segmentation

## สรุปใน 1 นาที

- ทดสอบ YOLO26s-Seg, YOLO11s-Seg และ YOLOv8s-Seg สำหรับ Person instance segmentation
- MOTS20 2,862 frames และ 26,894 Person GT instances ใช้ pretrained / no fine-tuning ภายใต้ controlled benchmark เดียวกัน
- Accuracy สูงสุด: YOLO26s-Seg; AP75 สูงสุด: YOLO26s-Seg; Recall สูงสุด: YOLO26s-Seg
- Inference เร็วสุด: YOLOv8s-Seg; pipeline เร็วสุดและ FPS สูงสุด: YOLO26s-Seg
- Peak allocated VRAM ต่ำสุด: YOLO26s-Seg
- คู่ที่ใกล้ที่สุดด้าน Mask mAP50-95: YOLO11s-Seg / YOLOv8s-Seg ต่าง 0.000505 — near-tied descriptively; ไม่ได้ทดสอบ statistical significance
- สถานะ PASS WITH WARNINGS; pipeline FPS ไม่รวม RLE preparation และ disk I/O

## ผลหลัก

| Model | Mask mAP50-95 | Recall | F1 | Inference ms | Pipeline ms | FPS | Peak VRAM MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s-Seg | 0.536640 | 0.789581 | 0.837161 | 17.308 | 78.828 | 12.686 | 872.67 |
| YOLO11s-Seg | 0.483269 | 0.764148 | 0.812887 | 15.693 | 88.836 | 11.257 | 1007.50 |
| YOLOv8s-Seg | 0.482763 | 0.767309 | 0.814124 | 15.220 | 92.173 | 10.849 | 1139.01 |


## แต่ละโมเดลเด่นด้านไหน

**YOLO26s-Seg**: นำ mAP50-95, AP75, Recall และ TP-only IoU/Dice พร้อมกัน โดย Recall 78.96% ยังไม่ครอบคลุม GT ทั้งหมด แม้ forward ช้าสุด (17.308 ms) แต่ pipeline เร็วสุด (78.828 ms) และ allocated VRAM ต่ำสุด (872.67 MiB) จึงเด่นเมื่อพิจารณาผลของ pipeline ที่วัดร่วมกับ accuracy และ memory

**YOLO11s-Seg**: mAP 0.483269 ใกล้ YOLOv8s (0.482763); AP75 และ matched-mask quality สูงกว่าเล็กน้อย แต่ Recall ต่ำกว่า ขณะที่ pipeline และ VRAM ดีกว่า YOLOv8s การมี parameters ต่ำสุดไม่ได้ทำให้ forward หรือ VRAM เป็นผู้ชนะ และไม่ควรใช้ mAP ทศนิยมท้าย ๆ ตัดสินความเหนือกว่า

**YOLOv8s-Seg**: forward เร็วสุด 15.220 ms และ Recall สูงกว่า YOLO11s แต่ pipeline ช้าสุด 92.173 ms และ VRAM สูงสุด 1139.01 MiB จึงเป็น candidate เมื่อเน้นเวลา forward โดยตรง; การใช้ segmentation pipeline ที่วัดครบขั้นต้องพิจารณาอันดับ pipeline แยกต่างหาก

## สิ่งที่น่าสนใจจากรอบนี้

- Observation: YOLO26s-Seg มี Mask mAP50-95 สูงสุด 0.536640; ห่างอันดับถัดไป 0.053371 บนสเกล 0–1
- Observation: YOLO26s-Seg นำ AP75 และ YOLO26s-Seg นำ Recall
- Observation: forward เร็วสุดคือ YOLOv8s-Seg, pipeline เร็วสุดและ FPS สูงสุดคือ YOLO26s-Seg, VRAM ต่ำสุดคือ YOLO26s-Seg
- คู่ที่ใกล้ที่สุดด้าน Mask mAP50-95: YOLO11s-Seg / YOLOv8s-Seg ต่าง 0.000505 — near-tied descriptively; ไม่ได้ทดสอบ statistical significance
- Interpretation: การเลือกต้องแยก accuracy, forward, pipeline และ memory ไม่สรุปว่า parameters ต่ำกว่าจะเร็วหรือใช้ VRAM ต่ำกว่าเสมอ

## Trade-off ที่เห็น

### Accuracy

YOLO26s นำ mAP50-95, AP75 และ Recall; YOLO11s/YOLOv8s near-tied เชิงพรรณนาที่ mAP แต่ AP75 กับ Recall เรียงต่างกัน ไม่มีการทดสอบนัยสำคัญ

### Speed

YOLOv8s forward เร็วสุด 15.220 ms แต่ YOLO26s pipeline เร็วสุด 78.828 ms อันดับกลับกันเมื่อรวม postprocessing ซึ่งมี mean YOLO26s / YOLO11s / YOLOv8s เท่ากับ 59.810 / 71.443 / 75.255 ms ตามลำดับ นี่เป็นการแจกแจงเวลาที่วัดได้ ไม่ใช่ข้อพิสูจน์เหตุเชิง architecture

### Memory / Resource

YOLO26s allocated VRAM ต่ำสุด แม้ YOLO11s มี parameters ต่ำสุด จำนวน parameters/GFLOPs ไม่ใช้แทนผลวัด memory หรือ latency

### ภาพรวม

YOLO26s เป็นตัวนำเมื่อเน้น accuracy ร่วมกับ pipeline และ memory ใน protocol นี้; YOLOv8s เป็นตัวนำด้าน forward ไม่สร้าง weighted score และยังไม่ยืนยันผล CCTV robustness

## สิ่งที่ต้องระวังในการตีความ

ผลนี้เป็น Person instance segmentation รายเฟรมบน MOTS20 ไม่ใช่ MOTS tracking; 26,894 GT instances เป็น annotation รายเฟรม ไม่ใช่จำนวนคนไม่ซ้ำ TP-only IoU/Dice พิจารณาเฉพาะคู่ที่ match ได้ ภาพต่อเนื่องสัมพันธ์กันและไม่มีการทดสอบ statistical significance ผลยังไม่ยืนยัน blur, low-light, มุมกล้อง, ระดับ occlusion หรือ deployment suitability จึงใช้เพื่อเลือก candidate for later CCTV robustness evaluation เท่านั้น

พบคำเตือน CPU NNPACK ใน log Small; ไม่พบ pycocotools DeprecationWarning ใน log รอบนี้ และแยก pipeline FPS ออกจากระบบที่บันทึก masks ครบวงจร

## ข้อมูลสำหรับนำไปรวมต่อ

[metrics/TIER_RESULTS.csv](metrics/TIER_RESULTS.csv) · [REPORT.md](REPORT.md) · [PRESENTATION_SUMMARY_TH.md](PRESENTATION_SUMMARY_TH.md) · [Master Study](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study)
