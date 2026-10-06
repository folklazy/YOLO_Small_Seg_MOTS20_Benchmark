# โพรโทคอล Small (S) — YOLO Instance Segmentation บน MOTS20

## 1. สถานะและหลักฐานต้นฉบับ

รอบ `benchmark-20261005T051531Z` เสร็จครบ 3 โมเดล ใช้โพรโทคอลที่กำหนดไว้ก่อนการทดลอง เอกสารนี้จัดรูปแบบภาษาไทยจากค่าที่ใช้จริง ไม่เปลี่ยนการตั้งค่าหรือคำนวณผลใหม่
[โพรโทคอลเดิมพร้อมบันทึก preflight](reports/archive/EXPERIMENT_PROTOCOL.md.pre-thai-language-20261006T071728Z) คง byte เดิมสำหรับตรวจค่า hash ใน STANDARDIZATION.json และหลักฐานการตรึงรอบทดลอง

## 2. โมเดลและข้อมูล

โมเดลตามลำดับแสดงผล: `yolo26s-seg.pt`, `yolo11s-seg.pt`, `yolov8s-seg.pt` ใช้ pretrained ไม่ฝึกเพิ่มไม่ปรับจูนและไม่ปรับตัวตามข้อมูลการทดสอบเทียบรุ่นที่มีให้ใช้ มิได้ทำให้ความจุหรือการฝึกเดิมเท่ากัน
ข้อมูลจาก `datasets/MOTS/MOTS/train/` เฉพาะ MOTS20-02, MOTS20-05, MOTS20-09, MOTS20-11 ตามลำดับ แล้วเรียงเฟรมขึ้น รวม 2,862 เฟรมและ Person GT รายเฟรม 26,894 instances GT อยู่ใน `gt/gt.txt`; Person class 2, ignore class 10 ตรวจภาพ มิติ ลำดับและ hash ตาม manifest เดิม ไม่แก้ข้อมูลและไม่สร้าง split ใหม่

## 3. การเตรียมภาพและ inference

ใช้ adapter และตัวประเมินที่ตรึงจาก Largest ตาม [หลักฐานต้นทาง](manifests/STANDARDIZATION.json) ภาพ OpenCV BGR ปรับขนาดเชิงเส้นโดยรักษาสัดส่วน เติม letterbox ตรงกลางเป็นสี่เหลี่ยมด้วยค่า 114: auto=False, scaleup=True, rect=False ไม่ยืดภาพ แปลงเป็น RGB BCHW FP32 /255 รูปร่าง 1×3×640×640
ใช้ CUDA:0 ตามข้อจำกัดการมองเห็น GPU เดิม, batch 1, imgsz 640, FP32, augment=False, retina_masks=True, agnostic_nms=False YOLO26 ใช้ nms=True เพื่อเลือกเส้นทาง one-to-many ก่อน fusion; backend end2end=False และ Person ของโมเดลคือ class 0 (`person`)
AP confidence floor 0.001 (framework NMS ใช้ >0.001), fixed confidence ≥0.25, NMS box IoU 0.70, model max_det=1000 ถ้าตัวเลือกถึงเพดานหรือเกิด NMS truncation ให้หยุด ห้ามเปลี่ยนนโยบายเฉพาะโมเดลสร้าง mask binary ที่มิติต้นฉบับและกรอง mask ว่าง ใช้การบีบอัด bit และส่ง CPU แบบเดิม บันทึกไม่สูญเสียข้อมูล COCO RLE พร้อม class, confidence, bbox xyxy และมิติต้นฉบับ

## 4. ตัวประเมินและนโยบาย ignore

ใช้ frozen ตัวชี้วัด.py และ mots.py ตาม hash เดิม ไม่แก้นิยามการจับคู่ fixed ตัวชี้วัดเรียง confidence ลงโดยคงลำดับเมื่อเท่ากัน จับคู่หนึ่งต่อหนึ่งแบบ greedy ที่ mask IoU ≥0.50 เลือก IoU สูงสุด แล้ว GT object ID เมื่อเสมอ ให้ valid Person GT มาก่อน เฉพาะ prediction ที่ไม่มีคู่จึงตรวจพื้นที่ทับ union ของ class 10 เทียบพื้นที่ prediction: IOA ≥0.50 ให้ ignore มิฉะนั้นเป็น FP ไม่ตัด Person mask ตาม ignore
TP/FP/FN รวมใช้คำนวณ Precision/Recall/F1 แบบ micro; TP-only IoU/Dice ใช้เฉพาะคู่ที่ผ่านการจับคู่จึงไม่รวม GT ที่พลาด AP เป็น Person-only COCO-style segmentation รายเฟรม: mask IoU 0.50:0.05:0.95, 101 จุด Recall และการเรียง confidence ค่า AP รวมมาจากข้อมูลทั้งหมด ไม่ใช่เฉลี่ย AP รายลำดับภาพ ไม่มี MOTS tracking ตัวชี้วัด

## 5. preflight และ maxDet

ใช้ 100 เฟรมเดิม: 25 เฟรมต่อ sequence ตาม frozen manifest และทดสอบทุกโมเดลก่อน accuracy ตรวจ AP maxDet 100/200/300/1000 โดย 1000 เป็นค่าอ้างอิง ความต่างสัมบูรณ์ AP50/AP75/mAP50-95 ต้อง <0.0001 ทุกโมเดล
ผลที่ตรึงใช้ AP maxDet=200 ซึ่งแยกจาก model max_det=1000 ตาม [ผล preflight](metrics/PREFLIGHT_MAXDET.csv) หาก 200 ไม่ผ่านต้องหยุดก่อน accuracy ไม่ลดการตั้งค่าเพราะโมเดลเล็กหรือปรับค่าเกณฑ์ให้ได้เปรียบ บันทึกการตัดสินใจและเวลาเดิมอยู่ในโพรโทคอลต้นฉบับ

## 6. การวัดเวลาและ VRAM

ใช้ 100 เฟรมตาม timing manifest เดิม 10 warmups และ 3 รอบที่ไม่ถูกรบกวนต่อโมเดล รวม 300 observations ต่อโมเดล รักษาลำดับ seeded/cyclic เดิม seed 20260929 ใช้โมเดลทีละตัว แยกเวลาโหลด ล้าง cache หลังปล่อยโมเดลเก่าและ synchronize CUDA ที่ขอบเขตแต่ละ stage
preprocessing รวมแปลงภาพ/H2D; inference คือ forward; postprocessing รวม NMS, mask native, binary validation, bit packing, ส่ง CPU และสร้าง prediction objects; pipeline เป็นผลรวมสาม stage วัด RLE preparation แยก ไม่รวมโหลด/อ่านและ decode ภาพ, GT, evaluator, visualization หรือ serialization ไม่บวก Ultralytics-inclusive diagnostic ซ้ำ
สรุปค่าเฉลี่ย, median/P50, population std และ P95 จาก observations ที่สะอาด; FPS=1000/ค่าเฉลี่ย pipeline ms รีเซ็ต allocator peak หลัง warmup และรวมโมเดลที่อยู่ในหน่วยความจำรายงาน peak allocated/reserved เป็น MiB โดยค่าหลักใช้ allocated peak สูงสุดจากรอบที่ยอมรับ ไม่ใช่หน่วยความจำทั้ง GPU จาก nvidia-smi

## 7. การรบกวนและข้อผิดพลาด

ตรวจ GPU ก่อน/ระหว่าง/หลังการวัด หากมี compute PID อื่น หรือ idle utilization ก่อนโหลด >5% ให้ถือว่าทั้งรอบถูกรบกวน เก็บหลักฐานและตัดออกจากผลหลัก ทำซ้ำได้เฉพาะส่วนที่ขาดด้วยการตั้งค่าเดิมเมื่อ GPU ว่างและใช้ที่เก็บใหม่ ห้ามจัดอันดับเมื่อเวลาไม่ครบ
เมื่อ OOM, crash, ไฟล์เสีย, ข้อมูลขาด, hash เปลี่ยน หรือตัวเลือกเต็มเพดาน ให้บันทึกโมเดล/เฟรม/สถานะแล้วหยุด ไม่ลด settings เฉพาะโมเดลไม่เขียนทับรอบเก่า การกลับมาทำงานต้องมีคำสั่งชัดเจนและตรวจ manifest/config/checkpoint/เฟรมที่ไม่ซ้ำก่อน ผลของ framework อยู่ภายใน repository เจ้าของการทดลอง

## 8. เอกสารและขอบเขตการทำงาน

ใช้ template ปัจจุบันของ [การศึกษาหลัก](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study): README นำทาง, REPORT บันทึกเทคนิค, RESULTS สรุปตัวเลข, PRESENTATION วิเคราะห์ภาพ สร้างรายงานหลัง accuracy และเวลาครบทุกโมเดล ภาพสร้างได้จาก prediction ที่บันทึกไว้เท่านั้น ไม่รัน inference เพื่อเอกสาร ไม่ใช้คะแนนถ่วงน้ำหนักหรืออ้างนัยสำคัญ/ความเหนือกว่าสำหรับ CCTV ขั้นสุดท้าย
ทั้งห้าขนาดเสร็จแล้ว ไม่เริ่ม benchmark ใหม่หรือสังเคราะห์ผล 17 โมเดลโดยอัตโนมัติ CSV การตั้งค่า checkpoint ข้อมูลและแหล่งรันเดิมคงเดิม
