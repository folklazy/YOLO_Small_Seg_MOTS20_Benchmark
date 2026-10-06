# Small (S) — การทดสอบ YOLO Instance Segmentation บน MOTS20

## ภาพรวม

เปรียบเทียบการแยก Person เป็นราย instance บน MOTS20 ด้วยโมเดล pretrained โดยไม่ฝึกเพิ่มหรือปรับจูน ประเมินรายเฟรมไม่ใช่การติดตามคน เอกสารนี้ใช้แนะนำ repository และเชื่อมไปยังผลเชิงตัวเลขรายงานเทคนิคและการวิเคราะห์ภาพ

## โมเดลที่ทดสอบ

| ตระกูล | โมเดล | ขนาด |
| --- | --- | --- |
| YOLO26 | YOLO26s-Seg | Small (S) |
| YOLO11 | YOLO11s-Seg | Small (S) |
| YOLOv8 | YOLOv8s-Seg | Small (S) |

## สถานะการทดลอง

COMPLETE / PASS WITH WARNINGS — ครบ 3/3 โมเดล โมเดลละ 2,862 เฟรมและ Person GT รายเฟรม 26,894 instances
รอบทดลอง: `benchmark-20261005T051531Z` ใช้ผลที่บันทึกไว้ ไม่มีการรัน inference ใหม่เพื่อปรับเอกสาร

## ผลลัพธ์หลัก

| โมเดล | Mask mAP50-95 | Recall | F1 | Inference (ms) | Pipeline (ms) | FPS | Peak allocated VRAM (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s-Seg | 0.536640 | 0.789581 | 0.837161 | 17.308 | 78.828 | 12.686 | 872.67 |
| YOLO11s-Seg | 0.483269 | 0.764148 | 0.812887 | 15.693 | 88.836 | 11.257 | 1007.50 |
| YOLOv8s-Seg | 0.482763 | 0.767309 | 0.814124 | 15.220 | 92.173 | 10.849 | 1139.01 |

## เอกสารประกอบ

- [บทสรุปเชิงตัวเลข](RESULTS_SUMMARY_TH.md)
- [การวิเคราะห์ภาพและพฤติกรรมเชิงคุณภาพ](PRESENTATION_SUMMARY_TH.md)
- [รายงานเทคนิค](REPORT.md)
- [โพรโทคอลการทดลอง](EXPERIMENT_PROTOCOL.md)

## การนำทางในชุดการศึกษา

[Largest (X/E)](https://github.com/folklazy/YOLO_Large_Seg_MOTS20_Benchmark) | [Second-largest (L/C)](https://github.com/folklazy/YOLO_Second_Largest_Seg_MOTS20_Benchmark) | [Medium (M)](https://github.com/folklazy/YOLO_Medium_Seg_MOTS20_Benchmark) | [Small (S)](https://github.com/folklazy/YOLO_Small_Seg_MOTS20_Benchmark) | [Nano (N)](https://github.com/folklazy/YOLO_Nano_Seg_MOTS20_Benchmark) | [การศึกษาหลัก](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study)

## หลักฐานสำหรับตรวจสอบซ้ำ

[ค่าตัวชี้วัด](metrics/) · [การตั้งค่า](configs/) · [หลักฐานและแหล่งที่มา](manifests/) · [ภาพและกราฟ](outputs/) · [บันทึกย้อนหลัง](reports/archive/)
