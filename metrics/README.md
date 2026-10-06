# ค่าตัวชี้วัด Small (S)

## หน้าที่

TIER_RESULTS.csv, PER_SEQUENCE_RESULTS.csv, TIMING_SUMMARY.csv, MODEL_COMPLEXITY.csv และ PREFLIGHT_MAXDET.csv เป็นตารางมาตรฐานสำหรับรอบ `benchmark-20261005T051531Z` สร้างหลังทุกโมเดลและการวัดเวลาครบแล้ว โดยเก็บความละเอียดตัวเลขจากแหล่งรันเดิม

## การใช้งานและข้อควรระวัง

ใช้ [schema ร่วม](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study/blob/main/DATA_SCHEMA.md) อ่านคอลัมน์และหน่วย AP รวมไม่ใช่เฉลี่ย AP รายลำดับภาพ; VRAM หลักคือ allocated peak ส่วน reserved peak และสถิติละเอียดอยู่ในแหล่งวัดเวลา หลักฐานรายเฟรมและ prediction เก็บในเครื่อง ไม่เผยแพร่ทุกไฟล์โดยอัตโนมัติ ไม่คำนวณค่าที่เสร็จแล้วใหม่เพื่อปรับเอกสาร
