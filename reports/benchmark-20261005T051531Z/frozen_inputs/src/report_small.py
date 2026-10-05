"""Render canonical Small reports from saved CSV only; no model inference."""
from pathlib import Path
import sys,json,csv,re,itertools
from validate import ROOT,EXP,sha,write,now
MASTER=ROOT/'YOLO_Instance_Segmentation_MOTS20_Scaling_Study'
def rows(p):return list(csv.DictReader(p.open()))
def fmt(k,v):
 if k in ['model','family','checkpoint','sequence']:return str(v)
 if v in ('','NA',None):return 'NA'
 if k in ['parameters','loaded_parameters']:return f'{int(v):,}'
 return f"{float(v):.{3 if k in ['inference_ms_mean','pipeline_ms_mean','fps','gflops','rle_preparation_ms_mean'] else 2 if k in ['peak_allocated_vram_mib','checkpoint_mb'] else 6}f}"
COLS=list(zip('model mask_map50_95 ap50 ap75 precision recall f1 tp_iou_mean tp_dice_mean inference_ms_mean pipeline_ms_mean fps peak_allocated_vram_mib parameters gflops checkpoint_mb'.split(),['Model','Mask mAP50-95','AP50','AP75','Precision','Recall','F1','TP-only IoU','TP-only Dice','Inference ms','Pipeline ms','FPS','Peak VRAM MiB','Params','GFLOPs','Checkpoint MB']))
COMPACT=[COLS[i] for i in [0,1,5,6,9,10,11,12]]
def table(data,cols):return '| '+' | '.join(h for k,h in cols)+' |\n| '+' | '.join('---' for _ in cols)+' |\n'+''.join('| '+' | '.join(fmt(k,r.get(k,'')) for k,h in cols)+' |\n' for r in data)
def render(name,sections):
 template=(EXP/'configs/report_templates'/f'TIER_{name}_TEMPLATE.md').read_text()
 headings=re.findall(r'^## (.+)$',template,re.M)
 assert set(headings)==set(sections),(name,set(headings)^set(sections))
 first=template.splitlines()[0].replace('{{TIER}}','Small')
 text=first+'\n\n'+'\n\n'.join('## '+h+'\n\n'+sections[h] for h in headings)+'\n'
 (EXP/(name+'.md')).write_text(text)
 return text

def main(run):
 assert json.loads((EXP/'manifests/final_integrity.json').read_text())['status']=='PASS'
 data=rows(EXP/'metrics/TIER_RESULTS.csv');seq=rows(EXP/'metrics/PER_SEQUENCE_RESULTS.csv');assert len(data)==3
 winner=lambda k,low=False:(min if low else max)(data,key=lambda r:float(r[k]))
 acc=winner('mask_map50_95');ap75=winner('ap75');recall=winner('recall');speed=winner('inference_ms_mean',True);pipeline=winner('pipeline_ms_mean',True);memory=winner('peak_allocated_vram_mib',True)
 metrics=[('Highest Mask mAP50-95','mask_map50_95',False),('Highest AP75','ap75',False),('Highest Recall','recall',False),('Fastest inference','inference_ms_mean',True),('Fastest pipeline','pipeline_ms_mean',True),('Highest FPS','fps',False),('Lowest VRAM','peak_allocated_vram_mib',True)]
 winning_table='| Category | Model | Value |\n|---|---|---|\n'+''.join(f"| {name} | {winner(k,low)['model']} | {fmt(k,winner(k,low)[k])} |\n" for name,k,low in metrics)
 navigation=re.search(r'## Study Navigation\n\n(.+)',(MASTER/'templates/TIER_README_TEMPLATE.md').read_text()).group(1)
 masterlink=f'[Master Study](https://github.com/folklazy/{MASTER.name})'
 reports='\n'.join(f'- [{n}]({n})' for n in ['PRESENTATION_SUMMARY_TH.md','RESULTS_SUMMARY_TH.md','REPORT.md','EXPERIMENT_PROTOCOL.md'])
 model_table='| Family | Model | Tier |\n|---|---|---|\n'+''.join(f"| {r['family']} | {r['model']} | Small (S) |\n" for r in data)
 main_table=table(data,COLS);compact=table(data,COMPACT)
 render('README',{'Overview':'Pretrained YOLO Person instance segmentation on MOTS20 using the frozen study protocol. Three Small checkpoints were evaluated without training, fine-tuning or adaptation. This is frame-level segmentation, not MOTS tracking.', 'Models':model_table,'Experimental Status':f'PASS WITH WARNINGS — COMPLETE, run `{run}`. All three models completed 2,862 frames and three clean timing rounds each.','Main Result':compact,'Reports':reports,'Study Navigation':navigation,'Reproducibility':'[configs/](configs/) · [metrics/](metrics/) · [manifests/](manifests/)'})
 # Rankings support concise per-model strengths and limitations without inventing causal explanations.
 ranks={k:{r['model']:j+1 for j,r in enumerate(sorted(data,key=lambda r:float(r[k]),reverse=k=='mask_map50_95'))} for k in ['mask_map50_95','inference_ms_mean','pipeline_ms_mean','peak_allocated_vram_mib']}
 paragraphs=[]
 for r in data:
  m=r['model'];ar=ranks['mask_map50_95'][m];ir=ranks['inference_ms_mean'][m];pr=ranks['pipeline_ms_mean'][m];vr=ranks['peak_allocated_vram_mib'][m]
  wins=[name for name,k,low in metrics if winner(k,low)['model']==m]
  strength='จุดเด่น: '+(' / '.join(wins) if wins else 'เป็นทางเลือกเปรียบเทียบของตระกูลนี้ภายใต้ protocol เดียวกัน')
  weaknesses=[]
  if ar>1:weaknesses.append('Mask mAP50-95 ต่ำกว่าตัวนำ')
  if ir>1:weaknesses.append('forward ช้ากว่าตัวที่เร็วที่สุด')
  if vr>1:weaknesses.append('ใช้ peak allocated VRAM มากกว่าตัวที่ต่ำสุด')
  if not weaknesses:weaknesses.append('ผลยังจำกัดอยู่ที่ข้อมูลและฮาร์ดแวร์ชุดนี้')
  text=f"{strength} จุดที่ด้อยกว่า: {'; '.join(weaknesses)} อันดับเชิงตัวเลขในสามโมเดลคือ accuracy {ar}, inference speed {ir}, pipeline speed {pr} และ VRAM ต่ำ {vr} "
  if m==acc['model']:text+='เหมาะเริ่มพิจารณาเมื่อเน้น accuracy แล้วตรวจว่าค่า latency และ memory อยู่ในข้อจำกัดของงาน'
  elif m==speed['model']:text+='เหมาะพิจารณาเมื่อเวลา forward เป็นข้อจำกัด โดยยอมรับ accuracy ที่ลดลงจากตัวนำ'
  elif m==memory['model']:text+='เหมาะพิจารณาเมื่อจำกัด memory โดยดู accuracy และ latency ประกอบ'
  else:text+='ยังเป็นผลอ้างอิงของตระกูลนี้ แต่ควรเทียบข้อจำกัดกับตัวนำก่อนเลือกใช้'
  paragraphs.append((m,text))
 ordered=sorted(data,key=lambda r:float(r['mask_map50_95']),reverse=True);gap=float(ordered[0]['mask_map50_95'])-float(ordered[1]['mask_map50_95'])
 near=[]
 for k,desc in [('mask_map50_95','Mask mAP50-95'),('inference_ms_mean','inference mean'),('pipeline_ms_mean','pipeline mean')]:
  a,b=min(itertools.combinations(data,2),key=lambda pair:abs(float(pair[0][k])-float(pair[1][k])))
  d=abs(float(a[k])-float(b[k]));relative=d/min(float(a[k]),float(b[k]))
  text=f"คู่ที่ใกล้ที่สุดด้าน {desc}: {a['model']} / {b['model']} ต่าง {fmt(k,str(d))}"+(' ms' if 'ms' in k else '')
  if relative<.01:text+=' — near-tied descriptively'
  text+='; ไม่ได้ทดสอบ statistical significance'
  near.append(text)
 insights=[f"Observation: {acc['model']} มี Mask mAP50-95 สูงสุด {fmt('mask_map50_95',acc['mask_map50_95'])}; ห่างอันดับถัดไป {gap:.6f} บนสเกล 0–1",f"Observation: {ap75['model']} นำ AP75 และ {recall['model']} นำ Recall",f"Observation: forward เร็วสุดคือ {speed['model']}, pipeline เร็วสุดและ FPS สูงสุดคือ {pipeline['model']}, VRAM ต่ำสุดคือ {memory['model']}",near[0], 'Interpretation: การเลือกต้องแยก accuracy, forward, pipeline และ memory ไม่สรุปว่า parameters ต่ำกว่าจะเร็วหรือใช้ VRAM ต่ำกว่าเสมอ']
 sequence_notes=[]
 for r in data:
  ss=[s for s in seq if s['model']==r['model']];best=max(ss,key=lambda s:float(s['mask_map50_95']));worst=min(ss,key=lambda s:float(s['mask_map50_95']))
  sequence_notes.append(f"{r['model']}: strongest {best['sequence']} ({fmt('mask_map50_95',best['mask_map50_95'])}); weakest {worst['sequence']} ({fmt('mask_map50_95',worst['mask_map50_95'])}) by Mask mAP50-95.")
 rankings={s:[r['model'] for r in sorted([r for r in seq if r['sequence']==s],key=lambda r:float(r['mask_map50_95']),reverse=True)] for s in ['MOTS20-02','MOTS20-05','MOTS20-09','MOTS20-11']}
 overall_order=[r['model'] for r in ordered]
 changes=[f"{s}: {' > '.join(order)}" for s,order in rankings.items() if order!=overall_order]
 sequence_notes.append('Ranking changes relative to pooled AP: '+('; '.join(changes) if changes else 'none across the four sequences.'))
 timing_source=f'timing/{run}/clean_repetition/summary.csv'
 rawtim=rows(EXP/timing_source)
 rle_notes='; '.join(f"{r['model']}: {float(r['rle_preparation_ms_mean']):.3f} ms" for r in rawtim)
 warnings='CPU NNPACK unsupported-hardware warnings occurred during checkpoint complexity inspection; pycocotools emitted a NumPy copy-keyword DeprecationWarning. Evaluator regression passed. No package versions were changed to suppress warnings. Primary timing contains only nine clean runs. Pipeline excludes RLE preparation and disk I/O; it is not end-to-end mask-saving/CCTV throughput.'
 limitation='ผลนี้เป็น Person instance segmentation รายเฟรมบน MOTS20 ไม่ใช่ MOTS tracking; 26,894 GT instances เป็น annotation รายเฟรม ไม่ใช่จำนวนคนไม่ซ้ำ TP-only IoU/Dice พิจารณาเฉพาะคู่ที่ match ได้ ภาพต่อเนื่องสัมพันธ์กันและไม่มีการทดสอบ statistical significance ผลยังไม่ยืนยัน blur, low-light, มุมกล้อง, ระดับ occlusion หรือ deployment suitability จึงใช้เพื่อเลือก candidate for later CCTV robustness evaluation เท่านั้น'
 sources='\n'.join(f'- [{n}](metrics/{n})' for n in ['TIER_RESULTS.csv','PER_SEQUENCE_RESULTS.csv','TIMING_SUMMARY.csv','MODEL_COMPLEXITY.csv','PREFLIGHT_MAXDET.csv'])
 compatibility='| Item | Status |\n|---|---|\n'+''.join(f'| {k} | PASS |\n' for k in ['Dataset','Evaluator','Preprocessing','Input size','Precision','Thresholds','maxDet','Timing protocol','Environment'])
 render('REPORT',{'1. Experiment Status':f'PASS WITH WARNINGS\n\n- Models completed: 3/3\n- Frames: 2,862 per model; Person GT instances: 26,894\n- Run ID: `{run}`', '2. Models Tested':table(data,[('family','Family'),('model','Model'),('parameters','Parameters'),('gflops','GFLOPs'),('checkpoint_mb','Checkpoint MB')]),'3. Protocol Compatibility':compatibility+'\nDataset compatibility: PASS\n\nPreprocessing compatibility: PASS\n\n[Common methodology](https://github.com/folklazy/'+MASTER.name+'/blob/main/METHODOLOGY_REFERENCE.md) · [Frozen protocol](EXPERIMENT_PROTOCOL.md)','4. Overall Results':main_table,'5. Tier Winners':winning_table,'6. Key Findings':'\n'.join('- '+x for x in insights),'7. Per-sequence Observations':'\n'.join('- '+x for x in sequence_notes),'8. Efficiency and Resource Observations':'\n'.join('- '+x for x in near[1:])+f"\n\n{memory['model']} has the lowest peak allocated VRAM. Loaded/fused parameters, GFLOPs and separate load times are retained in MODEL_COMPLEXITY.csv. Peak reserved VRAM is preserved in [source timing summary]({timing_source}). Separate RLE preparation means: {rle_notes}.",'9. Warnings and Anomalies':warnings,'10. Limitations':limitation,'11. Reproducibility and Source Artifacts':sources+f'\n\n[Standardization provenance](manifests/STANDARDIZATION.json) · [Final integrity](manifests/final_integrity.json) · [Timing source]({timing_source}) · [Plots](outputs/plots/INDEX.md)\n\nLossless per-frame RLE predictions and full telemetry remain local under predictions/{run}/ and timing/{run}/. Published manifests record hashes; no inference rerun is required to regenerate metrics.','12. Relation to Full Scaling Study':masterlink+' — Small only. Nano and final Master synthesis were not run.','Qualitative Analysis':'Saved lossless predictions support later qualitative reconstruction. This request uses the explicit presentation heading structure; no additional inference is needed.'})
 bullets=["ทดสอบ YOLO26s-Seg, YOLO11s-Seg และ YOLOv8s-Seg สำหรับ Person instance segmentation",'MOTS20 2,862 frames และ 26,894 Person GT instances ใช้ pretrained / no fine-tuning ภายใต้ controlled benchmark เดียวกัน',f"Accuracy สูงสุด: {acc['model']}; AP75 สูงสุด: {ap75['model']}; Recall สูงสุด: {recall['model']}",f"Inference เร็วสุด: {speed['model']}; pipeline เร็วสุดและ FPS สูงสุด: {pipeline['model']}",f"Peak allocated VRAM ต่ำสุด: {memory['model']}",near[0],'สถานะ PASS WITH WARNINGS; pipeline FPS ไม่รวม RLE preparation และ disk I/O']
 trade='### Accuracy\n\n'+f"{acc['model']} มี Mask mAP50-95 สูงสุด ต้องดู Recall และ AP75 ประกอบตามข้อจำกัด\n\n### Speed\n\n{speed['model']} มี inference mean ต่ำสุด ส่วน {pipeline['model']} มี pipeline mean ต่ำสุด; "+near[1]+'\n\n### Memory / Resource\n\n'+f"{memory['model']} ใช้ peak allocated VRAM ต่ำสุด จำนวน parameters และ GFLOPs ไม่ใช่ข้อพิสูจน์เหตุเชิงสาเหตุของ latency\n\n### ภาพรวม\n\nเลือกตามข้อจำกัดจริงโดยแยก accuracy, forward, pipeline และ memory ไม่สร้าง weighted score และไม่สรุปความพร้อมใช้งาน CCTV จากชุดนี้เพียงชุดเดียว"
 details=f'[REPORT.md](REPORT.md) · [metrics/TIER_RESULTS.csv](metrics/TIER_RESULTS.csv) · [RESULTS_SUMMARY_TH.md](RESULTS_SUMMARY_TH.md) · {masterlink}'
 cautions=limitation+'\n\nคงคำเตือน NNPACK / pycocotools ตามหลักฐาน และแยก pipeline FPS ออกจากระบบที่บันทึก masks ครบวงจร'
 render('RESULTS_SUMMARY_TH',{'สรุปใน 1 นาที':'\n'.join('- '+x for x in bullets),'ผลหลัก':compact,'แต่ละโมเดลเด่นด้านไหน':'\n\n'.join('**'+m+'**: '+text for m,text in paragraphs),'สิ่งที่น่าสนใจจากรอบนี้':'\n'.join('- '+x for x in insights),'Trade-off ที่เห็น':trade,'สิ่งที่ต้องระวังในการตีความ':cautions,'ข้อมูลสำหรับนำไปรวมต่อ':'[metrics/TIER_RESULTS.csv](metrics/TIER_RESULTS.csv) · [REPORT.md](REPORT.md) · [PRESENTATION_SUMMARY_TH.md](PRESENTATION_SUMMARY_TH.md) · '+masterlink})
 recommendation='| Priority | Recommended model | Reason |\n|---|---|---|\n'+f"| Accuracy | {acc['model']} | Mask mAP50-95 สูงสุด |\n| Speed | {speed['model']} (inference); {pipeline['model']} (pipeline) | เลือกตามส่วนที่เป็นข้อจำกัด |\n| Low VRAM | {memory['model']} | peak allocated VRAM ต่ำสุด |\n| Balanced trade-off | {acc['model']} เป็นตัวเริ่มพิจารณาแบบเน้น accuracy | ใช้เมื่อยอมรับ latency และ VRAM ในตารางได้; ไม่มีผู้ชนะสมดุลแบบไม่ขึ้นกับข้อจำกัด และไม่มีคะแนนรวม |\n"
 presentation=render('PRESENTATION_SUMMARY_TH',{'สรุปแบบกระชับ':'\n'.join('- '+x for x in bullets),'โมเดลที่ทดสอบ':'\n'.join(f"{j}. {r['family']}: `{r['checkpoint']}`" for j,r in enumerate(data,1)),'1. ผลลัพธ์หลัก':f"{acc['model']} มี Mask mAP50-95 สูงสุด {fmt('mask_map50_95',acc['mask_map50_95'])} ส่วน {speed['model']} มี inference mean ต่ำสุด และ {pipeline['model']} มี pipeline mean ต่ำสุด/ FPS สูงสุด\n\n{memory['model']} ใช้ peak allocated VRAM ต่ำสุด การเลือกต้องแยกเป้าหมาย accuracy, เวลา forward, pipeline และ memory ไม่ตีความความต่างเล็กมากเป็นความเหนือกว่าทางสถิติ",'2. ผลรวมโมเดล':table(data,COLS[:12]+[('peak_allocated_vram_mib','Peak VRAM allocated (MiB)'),('parameters','Parameters')]),'3. สรุปผลจากตาราง':'\n\n'.join('### '+m+'\n\n'+text for m,text in paragraphs),'4. Insight ที่สำคัญ':'\n'.join('- '+x for x in insights),'5. Trade-off':trade,'6. ถ้าต้องเลือกจาก Tier นี้':recommendation,'7. ข้อควรระวังในการตีความ':cautions,'รายละเอียดเต็ม':details})
 # Plots contain only saved canonical measurements.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=EXP/'outputs/plots';out.mkdir(parents=True,exist_ok=True);colors=['#2563eb','#ea580c','#9333ea'];names=[r['model'] for r in data]
 def bar(name,key,ylabel):
  fig,ax=plt.subplots(figsize=(8,4.8));vals=[float(r[key]) for r in data];ax.bar(names,vals,color=colors);ax.set_ylabel(ylabel);ax.set_title('Small (S) — MOTS20');ax.set_ylim(0,max(vals)*1.18)
  for j,r in enumerate(data):ax.text(j,vals[j],fmt(key,r[key]),ha='center',va='bottom',fontsize=9)
  fig.tight_layout();fig.savefig(out/name,dpi=160);plt.close(fig)
 bar('01_mask_map50_95.png','mask_map50_95','Mask mAP50-95 (0–1)');bar('03_inference_latency.png','inference_ms_mean','Inference mean (ms/frame)');bar('04_pipeline_fps.png','fps','Pipeline FPS (excludes RLE / disk I/O)');bar('05_peak_vram.png','peak_allocated_vram_mib','Peak allocated VRAM (MiB)')
 fig,ax=plt.subplots(figsize=(8,4.8));xs=range(3)
 for offset,key,col in [(-.2,'ap50','#2563eb'),(.2,'ap75','#ea580c')]:ax.bar([j+offset for j in xs],[float(r[key]) for r in data],width=.4,label=key.upper(),color=col)
 ax.set_xticks(list(xs),names);ax.set_ylim(0,1);ax.set_ylabel('AP (0–1)');ax.set_title('Small (S) — MOTS20');ax.legend();fig.tight_layout();fig.savefig(out/'02_ap50_ap75.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5.2))
 for j,r in enumerate(data):ax.scatter(float(r['inference_ms_mean']),float(r['mask_map50_95']),color=colors[j],label=r['model'],s=80)
 ax.set_xlabel('Inference mean (ms/frame)');ax.set_ylabel('Mask mAP50-95 (0–1)');ax.set_title('Small (S) — MOTS20');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'06_accuracy_vs_latency.png',dpi=160);plt.close(fig)
 plots=sorted(out.glob('0*.png'));(out/'INDEX.md').write_text('# Canonical Small plots\n\nSource: [TIER_RESULTS.csv](../../metrics/TIER_RESULTS.csv). Fixed model order; preserved measurements only.\n\n'+'\n'.join(f'- [{f.name}]({f.name})' for f in plots)+'\n')
 write(EXP/'manifests/PLOT_PROVENANCE.json',{'source':'metrics/TIER_RESULTS.csv','source_sha256':sha(EXP/'metrics/TIER_RESULTS.csv'),'generator':'src/report_small.py','generator_sha256':sha(Path(__file__)),'timestamp':now(),'outputs':{str(f.relative_to(EXP)):sha(f) for f in plots}})
 # Verify rendered table cells and local document links.
 cells=0;heading_map={h:k for k,h in COLS};heading_map.update({'Family':'family','Parameters':'parameters','Peak VRAM allocated (MiB)':'peak_allocated_vram_mib'})
 for doc in ['README.md','REPORT.md','RESULTS_SUMMARY_TH.md','PRESENTATION_SUMMARY_TH.md']:
  text=(EXP/doc).read_text();assert '{{' not in text;header=None
  for line in text.splitlines():
   if not line.startswith('|'):header=None;continue
   values=[x.strip() for x in line.strip('|').split('|')]
   if 'Model' in values and any(x in values for x in ['Mask mAP50-95','Parameters']):header=values;continue
   if header and all(re.fullmatch('[-:]+',v) for v in values):continue
   if header:
    model=values[header.index('Model')];r=next(r for r in data if r['model']==model)
    for h,v in zip(header,values):
     if h in heading_map:
      k=heading_map[h];assert v==fmt(k,r[k]),(doc,h,v);cells+=k not in ['model','family']
  for link in re.findall(r'\]\(([^)]+)\)',text):
   if not link.startswith(('http:','https:','#')):assert (EXP/link.split('#')[0]).exists(),(doc,link)
  expected=re.findall(r'^## .+$',(EXP/'configs/report_templates'/f'TIER_{doc.removesuffix(".md")}_TEMPLATE.md').read_text(),re.M)
  actual=re.findall(r'^## .+$',text,re.M)
  assert actual==expected,(doc,actual,expected)
 write(EXP/'manifests/DOCUMENT_VALIDATION.json',{'timestamp':now(),'status':'PASS','numeric_table_cells_verified':cells,'canonical_heading_order':'PASS','local_links':'PASS','presentation_policy':'Neutral tier summary; no mentor-specific closing section, per explicit Small user instruction.','document_sha256':{n:sha(EXP/n) for n in ['README.md','REPORT.md','RESULTS_SUMMARY_TH.md','PRESENTATION_SUMMARY_TH.md']},'template_sha256':{p.name:sha(p) for p in (EXP/'configs/report_templates').glob('TIER_*TEMPLATE.md')}})
 print('Reports, summaries, plots and document validation complete:',cells,'verified numeric table cells.')
if __name__=='__main__':main(sys.argv[1])
