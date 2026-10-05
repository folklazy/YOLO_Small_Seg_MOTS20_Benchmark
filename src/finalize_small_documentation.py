"""Final editorial review from completed canonical Small CSVs; no inference."""
from pathlib import Path
import csv, hashlib, json, re, datetime
EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open()))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def replace_section(doc,heading,body):
 s=doc.read_text();pattern=r'(?m)(^## '+re.escape(heading)+r'\n)[\s\S]*?(?=^## |\Z)'
 s,n=re.subn(pattern,lambda m:m[1]+'\n'+body.strip()+'\n\n',s);assert n==1,heading;doc.write_text(s)
def main():
 assert read(EXP/'manifests/SMALL_COMPLETE.json')['scientific_validation']=='PASS'
 guard={str(p.relative_to(EXP)):sha(p) for p in (EXP/'metrics').rglob('*') if p.is_file()}
 records=rows(EXP/'metrics/TIER_RESULTS.csv');d={r['model']:r for r in records};a,b,c=records
 times={(r['model'],r['stage']):r for r in rows(EXP/'metrics/TIMING_SUMMARY.csv')}
 before={n:sha(EXP/n) for n in ['README.md','REPORT.md','RESULTS_SUMMARY_TH.md','PRESENTATION_SUMMARY_TH.md','manifests/STANDARDIZATION.json']}
 claims=[]
 def value(r,k,dec=6,percent=False):
  v=float(r[k])*(100 if percent else 1);display=f'{v:.{dec}f}'+('%' if percent else '')
  claims.append({'model':r['model'],'source':'metrics/TIER_RESULTS.csv','field':k,'decimals':dec,'percent':percent,'display':display});return display
 def stage(r,k):
  display=f"{float(times[r['model'],k]['mean_ms']):.3f}"
  claims.append({'model':r['model'],'source':'metrics/TIMING_SUMMARY.csv','stage':k,'field':'mean_ms','decimals':3,'percent':False,'display':display});return display
 assert a['model']=='YOLO26s-Seg' and b['model']=='YOLO11s-Seg' and c['model']=='YOLOv8s-Seg'
 concise=[
  f"**{a['model']}**: นำ mAP50-95, AP75, Recall และ TP-only IoU/Dice พร้อมกัน โดย Recall {value(a,'recall',2,True)} ยังไม่ครอบคลุม GT ทั้งหมด แม้ forward ช้าสุด ({value(a,'inference_ms_mean',3)} ms) แต่ pipeline เร็วสุด ({value(a,'pipeline_ms_mean',3)} ms) และ allocated VRAM ต่ำสุด ({value(a,'peak_allocated_vram_mib',2)} MiB) จึงเด่นเมื่อพิจารณาผลของ pipeline ที่วัดร่วมกับ accuracy และ memory",
  f"**{b['model']}**: mAP {value(b,'mask_map50_95')} ใกล้ YOLOv8s ({value(c,'mask_map50_95')}); AP75 และ matched-mask quality สูงกว่าเล็กน้อย แต่ Recall ต่ำกว่า ขณะที่ pipeline และ VRAM ดีกว่า YOLOv8s การมี parameters ต่ำสุดไม่ได้ทำให้ forward หรือ VRAM เป็นผู้ชนะ และไม่ควรใช้ mAP ทศนิยมท้าย ๆ ตัดสินความเหนือกว่า",
  f"**{c['model']}**: forward เร็วสุด {value(c,'inference_ms_mean',3)} ms และ Recall สูงกว่า YOLO11s แต่ pipeline ช้าสุด {value(c,'pipeline_ms_mean',3)} ms และ VRAM สูงสุด {value(c,'peak_allocated_vram_mib',2)} MiB จึงเป็น candidate เมื่อเน้นเวลา forward โดยตรง; การใช้ segmentation pipeline ที่วัดครบขั้นต้องพิจารณาอันดับ pipeline แยกต่างหาก"
 ]
 replace_section(EXP/'RESULTS_SUMMARY_TH.md','แต่ละโมเดลเด่นด้านไหน','\n\n'.join(concise))
 detailed=[
  f"### {a['model']}\n\nนำทั้ง AP50, AP75, mAP50-95, Precision, Recall และ TP-only IoU/Dice ในชุดนี้ จึงมีหลักฐานเชิงตัวเลขหลายด้านร่วมกัน ไม่ได้อาศัย AP50 เพียงค่าเดียว Recall {value(a,'recall',2,True)} ยังเหลือ GT ที่พลาดหรือจับคู่ไม่ผ่าน; TP-only quality เป็นคุณภาพเฉพาะคู่ที่ผ่านการจับคู่และไม่อธิบายคนที่พลาด\n\nสิ่งที่แลกคือ forward {value(a,'inference_ms_mean',3)} ms ซึ่งช้าสุด แต่ postprocessing ที่วัด {stage(a,'postprocessing')} ms ต่ำกว่าอีกสองโมเดล ทำให้ pipeline {value(a,'pipeline_ms_mean',3)} ms เร็วสุดและ VRAM {value(a,'peak_allocated_vram_mib',2)} MiB ต่ำสุด หากงานใช้ pipeline ตามนิยามนี้ รุ่นนี้นำทั้ง accuracy, pipeline และ memory; หากข้อจำกัดอยู่ที่ forward อย่างเดียวต้องเทียบ YOLOv8s เพิ่ม",
  f"### {b['model']}\n\nmAP {value(b,'mask_map50_95')} ใกล้ YOLOv8s ({value(c,'mask_map50_95')}) มาก โดย AP75 {value(b,'ap75')} เทียบกับ {value(c,'ap75')} และ TP-only IoU/Dice สูงกว่าเล็กน้อย แต่ Recall {value(b,'recall',2,True)} ต่ำกว่า YOLOv8s ({value(c,'recall',2,True)}) จึงเป็นความต่างของ precision/คุณภาพคู่ที่ match กับความครบถ้วน มากกว่าจะสรุปว่าทุกด้านดีกว่า\n\nforward {value(b,'inference_ms_mean',3)} ms ช้ากว่า YOLOv8s เล็กน้อย แต่ pipeline {value(b,'pipeline_ms_mean',3)} ms และ VRAM {value(b,'peak_allocated_vram_mib',2)} MiB ต่ำกว่า YOLOv8s มี loaded parameters น้อยสุดในสามรุ่นแต่ไม่ได้ชนะด้านเวลาและ VRAM การจัดอันดับ mAP ระหว่างคู่นี้สลับในบาง sequence: YOLOv8s สูงกว่าใน MOTS20-02 และ MOTS20-11 ส่วน YOLO11s สูงกว่าใน MOTS20-05 และ MOTS20-09 จึงไม่ควรเลือกจาก pooled mAP ทศนิยมท้าย ๆ อย่างเดียว",
  f"### {c['model']}\n\nจุดเด่นคือ inference {value(c,'inference_ms_mean',3)} ms ต่ำสุดในกลุ่ม และ Recall {value(c,'recall',2,True)} สูงกว่า YOLO11s แม้ mAP จะใกล้กัน ขณะที่ AP75 และ TP-only matched-mask quality ต่ำกว่าเล็กน้อย คู่นี้จึงมี trade-off ที่คะแนนรวมใกล้กันอาจบดบัง; ยังไม่มีการทดสอบนัยสำคัญ\n\npostprocessing {stage(c,'postprocessing')} ms สูงสุดใน timing ที่วัด ทำให้ pipeline {value(c,'pipeline_ms_mean',3)} ms ช้าสุดแม้ forward เร็วสุด พร้อม VRAM {value(c,'peak_allocated_vram_mib',2)} MiB สูงสุด จึงควรใช้เป็นตัวเทียบเมื่อเน้น forward และเป็น baseline ของตระกูลเดิม ผลนี้ไม่พิสูจน์ว่า architecture รุ่นเก่าด้อยกว่าเสมอ เพราะ checkpoint และการฝึกเดิมไม่ได้ถูกควบคุมให้เหมือนกัน"
 ]
 replace_section(EXP/'PRESENTATION_SUMMARY_TH.md','3. สรุปผลจากตาราง','\n\n'.join(detailed))
 speed=f"YOLOv8s forward เร็วสุด {value(c,'inference_ms_mean',3)} ms แต่ YOLO26s pipeline เร็วสุด {value(a,'pipeline_ms_mean',3)} ms อันดับกลับกันเมื่อรวม postprocessing ซึ่งมี mean YOLO26s / YOLO11s / YOLOv8s เท่ากับ {stage(a,'postprocessing')} / {stage(b,'postprocessing')} / {stage(c,'postprocessing')} ms ตามลำดับ นี่เป็นการแจกแจงเวลาที่วัดได้ ไม่ใช่ข้อพิสูจน์เหตุเชิง architecture"
 for name,heading in [('RESULTS_SUMMARY_TH.md','Trade-off ที่เห็น'),('PRESENTATION_SUMMARY_TH.md','5. Trade-off')]:
  replace_section(EXP/name,heading,f"### Accuracy\n\nYOLO26s นำ mAP50-95, AP75 และ Recall; YOLO11s/YOLOv8s near-tied เชิงพรรณนาที่ mAP แต่ AP75 กับ Recall เรียงต่างกัน ไม่มีการทดสอบนัยสำคัญ\n\n### Speed\n\n{speed}\n\n### Memory / Resource\n\nYOLO26s allocated VRAM ต่ำสุด แม้ YOLO11s มี parameters ต่ำสุด จำนวน parameters/GFLOPs ไม่ใช้แทนผลวัด memory หรือ latency\n\n### ภาพรวม\n\nYOLO26s เป็นตัวนำเมื่อเน้น accuracy ร่วมกับ pipeline และ memory ใน protocol นี้; YOLOv8s เป็นตัวนำด้าน forward ไม่สร้าง weighted score และยังไม่ยืนยันผล CCTV robustness")
 # Embed retained CSV-derived plots using publishable repository-relative paths.
 p=EXP/'PRESENTATION_SUMMARY_TH.md';s=p.read_text()
 for heading,link,label in [('### Speed','outputs/plots/01_mask_map50_95.png','Small canonical Mask mAP50-95'),('### Memory / Resource','outputs/plots/04_pipeline_fps.png','Small canonical pipeline FPS'),('### ภาพรวม','outputs/plots/05_peak_vram.png','Small canonical peak allocated VRAM')]:
  s=s.replace(heading,'!['+label+']('+link+')\n\n'+heading,1)
 p.write_text(s)
 for name in ['README.md','REPORT.md','RESULTS_SUMMARY_TH.md','PRESENTATION_SUMMARY_TH.md']:
  p=EXP/name;s=p.read_text();s=s.replace('CPU NNPACK unsupported-hardware warnings occurred during checkpoint complexity inspection; pycocotools emitted a NumPy copy-keyword DeprecationWarning.','CPU NNPACK unsupported-hardware warnings were captured in this Small run. No pycocotools DeprecationWarning was captured in the Small logs; historical warnings from earlier tiers are not counted as new Small warnings.')
  s=s.replace('คงคำเตือน NNPACK / pycocotools ตามหลักฐาน','พบคำเตือน CPU NNPACK ใน log Small; ไม่พบ pycocotools DeprecationWarning ใน log รอบนี้')
  p.write_text(s)
 stdpath=EXP/'manifests/STANDARDIZATION.json';std=read(stdpath)
 std['notes']=[x.replace('PASS_WITH_WARNINGS retains NNPACK CPU inspection and pycocotools/NumPy deprecation warnings.','PASS_WITH_WARNINGS retains observed CPU NNPACK warnings. No pycocotools/NumPy DeprecationWarning was captured in the Small run logs.') for x in std['notes']];write(stdpath,std)
 validation=read(EXP/'manifests/DOCUMENT_VALIDATION.json')
 for name in validation['document_sha256']:
  doc=(EXP/name).read_text();template=(EXP/'configs/report_templates'/f'TIER_{name.removesuffix(".md")}_TEMPLATE.md').read_text()
  assert re.findall(r'^## .+$',doc,re.M)==re.findall(r'^## .+$',template,re.M)
  assert not re.search(r'(?i)mentor|อาจารย์|สรุปสำหรับคุยกับพี่',doc)
 validation['document_sha256']={n:sha(EXP/n) for n in validation['document_sha256']}
 validation['final_editorial_review']='Canonical table values unchanged; interpretations refined from TIER_RESULTS, PER_SEQUENCE_RESULTS and TIMING_SUMMARY; warning claims checked against Small logs.'
 write(EXP/'manifests/DOCUMENT_VALIDATION.json',validation)
 assert guard=={str(p.relative_to(EXP)):sha(p) for p in (EXP/'metrics').rglob('*') if p.is_file()}
 write(EXP/'manifests/FINAL_DOCUMENT_REVIEW.json',{'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS','generator_sha256':sha(Path(__file__)),'before':before,'after':{n:sha(EXP/n) for n in before},'canonical_metrics_sha256':{f'metrics/{n}.csv':sha(EXP/'metrics'/f'{n}.csv') for n in ['TIER_RESULTS','PER_SEQUENCE_RESULTS','TIMING_SUMMARY','MODEL_COMPLEXITY','PREFLIGHT_MAXDET']},'numeric_claims':claims,'measurements_changed':False,'inference_run':False,'warning_evidence':'NNPACK in runner_console.log; no DeprecationWarning in Small run logs'})
 print('Small final editorial review PASS; measured artifacts unchanged.')
if __name__=='__main__':main()
