"""Read-only scientific audit and canonical conversion of completed Small measurements."""
from pathlib import Path
import sys,json,csv,gzip,math,hashlib,shutil,importlib.metadata as md
import numpy as np
# Final script lives in src/. No inference imports or model construction.
from validate import ROOT,EXP,MODELS,SEQS,sha,now,write
SID='yolo_instance_segmentation_mots20_scaling'
MASTER=ROOT/'YOLO_Instance_Segmentation_MOTS20_Scaling_Study'
def read(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open()))
def csvout(p,cols,rr):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rr)
def label(cp):return cp.removesuffix('.pt').replace('yolov','YOLOv').replace('yolo','YOLO').replace('-seg','-Seg')
def family(cp):return next(k for k in ['YOLO26','YOLO11','YOLOv8'] if label(cp).startswith(k))
def identity(cp):return dict(study_id=SID,schema_version='1.0',experiment=EXP.name,tier='small',family=family(cp),model=label(cp))
def main(run):
 assert len(MODELS)==3 and MODELS==['yolo26s-seg.pt','yolo11s-seg.pt','yolov8s-seg.pt']
 import yaml
 cfg=yaml.safe_load((EXP/'configs/benchmark.yaml').read_text());frozen=read(EXP/'manifests'/f'{run}_full_freeze.json')
 reference=ROOT/'YOLO_Large_Seg_MOTS20_Benchmark'
 reference_cfg=yaml.safe_load((reference/'configs/benchmark.yaml').read_text())
 assert {k:v for k,v in cfg.items() if k!='models'}=={k:v for k,v in reference_cfg.items() if k!='models'}
 assert sha(EXP/'configs/benchmark.yaml')==frozen['config_sha256'] and sha(EXP/'EXPERIMENT_PROTOCOL.md')==frozen['protocol_sha256']
 assert all(sha(EXP/p)==h for p,h in frozen['sources'].items())
 archive=EXP/'reports'/run/'frozen_inputs';archive_index=read(EXP/'manifests'/f'{run}_input_archive.json')['hashes']
 assert all(sha(archive/p)==h for p,h in archive_index.items())
 assert all(sha(ROOT/v['original'])==v['sha256']==sha(ROOT/v['snapshot']) for v in read(EXP/'manifests/source_manifest.json').values())
 images=read(EXP/'manifests/images.json');dataset=read(EXP/'manifests/dataset_manifest.json')
 assert len(images)==2862 and images==read(reference/'manifests/images.json')
 assert all(sha(ROOT/r['path'])==r['sha256'] for r in images)
 for s in dataset['sequences']:
  d=ROOT/dataset['dataset_root']/s['sequence'];assert sha(d/'gt/gt.txt')==s['gt_sha256'] and sha(d/'seqinfo.ini')==s['seqinfo_sha256']
 for f in ['images.json','preflight_frames.json','timing_frames.json','visualization_frames.json']:assert sha(EXP/'manifests'/f)==sha(reference/'manifests'/f)
 import ultralytics
 for f,h in read(EXP/'manifests/framework_source_hashes.json').items():assert sha(Path(ultralytics.__file__).parent/f)==h
 env=read(EXP/'manifests/environment.json');assert all(md.version(k)==v for k,v in env['packages'].items())
 checkpoints={r['filename']:r for r in read(EXP/'manifests/checkpoint_manifest.json')};assert set(checkpoints)==set(MODELS)
 assert all(sha(ROOT/'models'/cp)==r['sha256'] for cp,r in checkpoints.items())
 out=EXP/'metrics'/run;timdir=EXP/'timing'/run/'clean_repetition'
 accuracy=rows(out/'per_model.csv');sequences=rows(out/'per_sequence.csv');timing=rows(timdir/'summary.csv');preflight=rows(out/'preflight_maxdet.csv')
 assert len(accuracy)==3 and len(sequences)==12 and len(timing)==3 and len(preflight)==12
 assert {r['model'] for r in accuracy}==set(MODELS)
 assert read(EXP/'manifests'/f'{run}_preflight.json')['selected_max_dets']==200
 assert all(r['converged']=='True' for r in preflight if r['max_dets']=='200')
 checks=[];prediction_hashes={};metadata={};audit_rows=[]
 expected=[f"{r['sequence']}_{r['frame']:06d}" for r in images]
 required=['map50_95','ap50','ap75','precision','recall','f1','matched_iou_mean','matched_dice_mean']
 for r in accuracy+sequences:
  assert all(math.isfinite(float(r[k])) for k in required)
  tp,fp,fn=(int(r[k]) for k in ['tp','fp','fn'])
  assert tp+fn==int(r['gt_persons']);assert tp+fp+int(r['ignored_predictions'])==int(r['predictions_at_confidence'])
  for k,value in [('precision',tp/(tp+fp)),('recall',tp/(tp+fn)),('f1',2*tp/(2*tp+fp+fn))]:assert math.isclose(float(r[k]),value,rel_tol=1e-12,abs_tol=1e-14)
 for cp in MODELS:
  d=EXP/'predictions'/run/'accuracy'/cp.removesuffix('.pt');meta=read(d/'metadata.json');metadata[cp]=meta
  assert meta['status']=='PASS' and meta['images_successful']==2862 and meta['images_failed']==0 and meta['frames']==expected
  assert meta['config']==cfg and meta['config_sha256']==frozen['config_sha256'] and meta['protocol_sha256']==frozen['protocol_sha256']
  assert meta['checkpoint_sha256']==checkpoints[cp]['sha256'] and meta['end2end'] is False and meta['precision']=='fp32'
  args=meta['effective_predictor_args'];assert args['nms'] is True and str(args['device'])=='0'
  assert len(list((d/'predictions').glob('*.json.gz')))==2862
  max_candidates=0;total_predictions=0
  for im in images:
   path=d/'predictions'/f"{im['sequence']}_{im['frame']:06d}.json.gz"
   with gzip.open(path,'rt') as f:r=json.load(f)
   assert r['model']==cp and (r['sequence'],r['frame'])==(im['sequence'],im['frame'])
   assert [r['width'],r['height']]==[im['width'],im['height']] and r['input_shape']==[1,3,640,640]
   assert r['post_nms_candidates']<1000
   max_candidates=max(max_candidates,r['post_nms_candidates']);total_predictions+=len(r['predictions'])
   for pred in r['predictions']:
    assert pred['class']==0 and math.isfinite(pred['confidence']) and pred['confidence']>.001
    assert all(math.isfinite(x) for x in pred['bbox_xyxy']) and pred['rle']['size']==[im['height'],im['width']]
   prediction_hashes[str(path.relative_to(EXP))]=sha(path)
  a=next(r for r in accuracy if r['model']==cp);ss=[r for r in sequences if r['model']==cp]
  assert len(ss)==4 and {r['sequence'] for r in ss}==set(SEQS)
  assert int(a['frames'])==2862 and int(a['gt_persons'])==26894 and int(a['predictions_ap_floor'])==total_predictions
  for k in ['frames','gt_persons','tp','fp','fn','ignored_predictions','predictions_ap_floor','predictions_at_confidence']:assert sum(int(r[k]) for r in ss)==int(a[k])
  audit_rows.append({'model':cp,'frames':2862,'status':'PASS','max_post_nms_candidates':max_candidates,'saved_predictions':total_predictions,'finite_outputs':True,'input_shape':[1,3,640,640]})
 # Timing tables are derived only from accepted clean observations; validate their measured order and pooled statistics.
 runs=read(timdir/'runs.json');assert len(runs)==9
 timing_expected=[(r['sequence'],r['frame']) for r in read(EXP/'manifests/timing_frames.json')]
 for cp in MODELS:
  accepted=[]
  for rnd in range(1,4):
   record=read(timdir/f'round{rnd}-{cp}.json');rr=rows(timdir/f'round{rnd}-{cp}.csv')
   assert record['status']=='PASS' and record['contaminated'] is False and len(rr)==100
   assert [(r['sequence'],int(r['frame'])) for r in rr]==timing_expected
   assert all(r['contaminated']=='False' for r in rr)
   accepted.extend(rr)
  t=next(r for r in timing if r['model']==cp);assert t['frames']=='300' and t['clean_rounds']=='3'
  for metric in ['preprocess_ms','inference_ms','postprocess_ms','ultralytics_postprocess_inclusive_ms','total_ms','rle_preparation_ms']:
   vals=np.asarray([float(r[metric]) for r in accepted]);assert np.isfinite(vals).all()
   calculated={'mean':vals.mean(),'median':np.median(vals),'std':vals.std(),'p50':np.percentile(vals,50),'p95':np.percentile(vals,95)}
   for k,v in calculated.items():assert math.isclose(float(t[metric+'_'+k]),float(v),rel_tol=1e-12,abs_tol=1e-12)
  assert math.isclose(float(t['fps']),1000/float(t['total_ms_mean']),rel_tol=1e-12)
  for mem in ['peak_gpu_allocated_mib','peak_gpu_reserved_mib']:
   assert float(t[mem])==max(r[mem] for r in runs if r['model']==cp)
 checks=[{'item':k,'status':'PASS'} for k in ['Exactly 3 expected models','2862 frames and 26894 GT per model','Dataset hashes, dimensions and ordering','Frozen evaluator and preprocessing','Config and protocol freeze','Framework and environment','AP maxDet 200 preflight','Finite native Person predictions','No model cap saturation','Accuracy arithmetic and sequence completeness','Three clean timing rounds per model','Exact timing frame order','Timing pooled statistics and VRAM','Checkpoint hashes','Frozen input archive']]
 csvout(out/'protocol_consistency.csv',['item','status'],checks)
 write(EXP/'manifests'/f'{run}_prediction_hashes.json',prediction_hashes)
 write(EXP/'manifests'/f'{run}_accuracy_integrity.json',{'status':'PASS','models':audit_rows,'no_inference_rerun':True})
 schema=read(MASTER/'schemas.json');mainrows=[];seqrows=[];time_rows=[];complexity=[];pf=[]
 mapping={'gt_instances':'gt_persons','mask_map50_95':'map50_95','tp_iou_mean':'matched_iou_mean','tp_dice_mean':'matched_dice_mean'}
 for cp in MODELS:
  b=identity(cp);a=next(r for r in accuracy if r['model']==cp);t=next(r for r in timing if r['model']==cp);ck=checkpoints[cp]
  r={k:a.get(mapping.get(k,k),'') for k in schema['schemas']['TIER_RESULTS']};r.update(b)
  r.update(checkpoint=cp,status='PASS_WITH_WARNINGS',inference_ms_mean=t['inference_ms_mean'],pipeline_ms_mean=t['total_ms_mean'],fps=t['fps'],peak_allocated_vram_mib=t['peak_gpu_allocated_mib'],parameters=str(ck['parameters_loaded']),gflops=str(ck['gflops_nms_unfused']),checkpoint_mb=str(ck['bytes']/1e6),ap_maxdet=str(cfg['ap_max_dets']),fixed_confidence=str(cfg['fixed_confidence']),ap_confidence_floor=str(cfg['ap_confidence_floor']),nms_iou=str(cfg['nms_iou']),evaluation_iou=str(cfg['matching_iou']),imgsz=str(cfg['imgsz']),precision_mode='FP32',device='CUDA:0',run_id=run,source_artifact=f'metrics/{run}/per_model.csv;timing/{run}/clean_repetition/summary.csv;manifests/checkpoint_manifest.json')
  mainrows.append(r)
  for a in sequences:
   if a['model']==cp:seqrows.append({**{k:a.get(mapping.get(k,k),'') for k in schema['schemas']['PER_SEQUENCE_RESULTS']},**b,'predictions':a['predictions_at_confidence']})
  for stage,prefix in [('preprocessing','preprocess'),('inference','inference'),('postprocessing','postprocess'),('pipeline','total'),('rle_preparation','rle_preparation'),('ultralytics_postprocess_inclusive','ultralytics_postprocess_inclusive')]:
   time_rows.append({**b,'stage':stage,**{k+'_ms':t[prefix+'_ms_'+k] for k in ['mean','median','std','p50','p95']},'repetitions':'3','measured_frames':'300','contamination_status':'CLEAN'})
  complexity.append({**b,'loaded_parameters':str(ck['parameters_loaded']),'fused_parameters':str(metadata[cp]['runtime_parameters']),'gflops':str(ck['gflops_nms_unfused']),'checkpoint_mb':str(ck['bytes']/1e6),'model_load_seconds':t['load_seconds_mean']})
  for a in preflight:
   if a['model']==cp:pf.append({**a,**b,'checkpoint':cp})
 for name,rr in [('TIER_RESULTS',mainrows),('PER_SEQUENCE_RESULTS',seqrows),('TIMING_SUMMARY',time_rows),('MODEL_COMPLEXITY',complexity)]:csvout(EXP/'metrics'/f'{name}.csv',schema['schemas'][name],rr)
 csvout(EXP/'metrics/PREFLIGHT_MAXDET.csv',schema['PREFLIGHT_MAXDET'],pf)
 # Canonical schemas have no reserved-VRAM field; preserve it with load/baseline details in accepted timing source.
 sources=[out/'per_model.csv',out/'per_sequence.csv',out/'preflight_maxdet.csv',out/'protocol_consistency.csv',timdir/'summary.csv',timdir/'runs.json',EXP/'configs/benchmark.yaml',EXP/'EXPERIMENT_PROTOCOL.md',EXP/'manifests/environment.json',EXP/'manifests/checkpoint_manifest.json',EXP/'manifests/source_manifest.json',EXP/'manifests/dataset_manifest.json',EXP/'manifests'/f'{run}_full_freeze.json',EXP/'manifests'/f'{run}_input_archive.json',EXP/'manifests'/f'{run}_prediction_hashes.json']
 for cp in MODELS:sources.append(EXP/'predictions'/run/'accuracy'/cp.removesuffix('.pt')/'metadata.json')
 source_hashes={str(p.relative_to(EXP)):sha(p) for p in sources}
 write(EXP/'manifests/STANDARDIZATION.json',{'study_id':SID,'schema_version':'1.0','repository':EXP.name,'experiment':EXP.name,'tier':'small','source_run_ids':[run],'source_artifact_paths':list(source_hashes),'source_artifact_sha256':source_hashes,'evaluator_hash':{f'src/frozen_pilot/{n}':sha(EXP/'src/frozen_pilot'/n) for n in ['metrics.py','mots.py']},'config_hash':frozen['config_sha256'],'protocol_hash':frozen['protocol_sha256'],'dataset_manifest_hash':sha(EXP/'manifests/dataset_manifest.json'),'standardization_timestamp':now(),'inference_rerun':False,'metrics_recalculated':False,'historical_documents_archived':read(EXP/'manifests/SMALL_SETUP.json')['archived_setup_documents'],'notes':['New Small experiment, not a rerun. Accuracy calculated once from saved lossless predictions. Canonical conversion preserves structured numeric strings.','metrics_recalculated=False refers to normalization; independent arithmetic/statistics verification does not replace measured results.','Primary timing uses only three clean rounds/model. Reserved VRAM and load/baseline records remain in source timing summary.','PASS_WITH_WARNINGS retains NNPACK CPU inspection and pycocotools/NumPy deprecation warnings.','Latest Small user request explicitly selects numerical summary/presentation headings; pinned report templates are in configs/report_templates. No mentor-specific content. Frozen measurement protocol is unchanged.']})
 write(EXP/'manifests/final_integrity.json',{'timestamp':now(),'status':'PASS','checks':checks,'frames_per_model':2862,'gt_instances':26894,'models':MODELS,'timing_clean_rounds':9,'inference_rerun':False,'frozen_runtime_sources_unchanged':True,'archive_unchanged':True,'dataset_and_checkpoint_hashes_unchanged':True,'environment_packages_unchanged':True})
 public={k:env[k] for k in ['timestamp','python','packages','torch_version','torchvision_version','cuda_runtime','cuda_available','CUDA_VISIBLE_DEVICES']};public.update(gpu='Tesla T4',driver='580.178.04',raw_environment_sha256=sha(EXP/'manifests/environment.json'));write(EXP/'manifests/environment_public.json',public)
 print('Small consistency PASS: 3 models, 8586 saved accuracy frames, 9 clean timing rounds; canonical CSVs exported.')
if __name__=='__main__':main(sys.argv[1])
