"""Gated, sequential benchmark; all accuracy is regenerated from saved RLE."""
from pathlib import Path
import os,sys,json,csv,gzip,datetime,time,gc,traceback,subprocess,logging,random,shutil
from contextlib import redirect_stdout,redirect_stderr
from validate import ROOT,EXP,SEQS,MODELS,sha,now,write,cmd
sys.path.insert(0,str(EXP/'src/frozen_pilot'))
import numpy as np
import torch,torchvision,cv2,yaml
from mots import load_frames
from metrics import CompactPrediction,fixed_metrics,ap_metrics,ratios
from benchmark_adapter import Adapter

def read(p): return json.loads(p.read_text())
def config(): return yaml.safe_load((EXP/'configs/benchmark.yaml').read_text())
def csvwrite(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(values):
    a=np.asarray(values,dtype=float)
    if not len(a): return {k:None for k in ['mean','median','std','p25','p75','p50','p95']}
    return {'mean':float(a.mean()),'median':float(np.median(a)),'std':float(a.std()),
            **{f'p{p}':float(np.percentile(a,p)) for p in [25,75,50,95]}}
def key(f): return f'{f.sequence}_{f.number:06d}'
def clean(): gc.collect();torch.cuda.empty_cache()
def gpu():
    return {'timestamp':now(),'state':cmd('nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem','--format=csv'),
            'processes':cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader')}
def contaminated(g,idle=False):
    if g['state']['returncode'] or g['processes']['returncode']: return True
    for line in g['processes']['stdout'].splitlines():
        if line.strip() and line.split(',')[0].strip()!=str(os.getpid()): return True
    if idle:
        try:
            if float(g['state']['stdout'].splitlines()[1].split(',')[5].strip().split()[0])>5:return True
        except (ValueError,IndexError):return True
    return False
def frames_for(sample=None):
    manifest=read(EXP/'manifests/images.json');fs=[]
    for s in SEQS:
        nums=[r['frame'] for r in (sample or manifest) if r['sequence']==s]
        fs.extend(load_frames(ROOT/'datasets/MOTS/MOTS/train',s,nums))
    return fs
def encode_preds(ps):
    return [{'class':p.class_id,'confidence':p.confidence,'bbox_xyxy':p.bbox_xyxy,
             'rle':{'size':p.rle['size'],'counts':p.rle['counts'].decode('ascii')}} for p in ps]
def decode_preds(rows):
    return [CompactPrediction(p['confidence'],p['class'],p['bbox_xyxy'],
                              {'size':p['rle']['size'],'counts':p['rle']['counts'].encode('ascii')}) for p in rows]
def save_prediction(path,record):
    assert not path.exists(),f'No overwrite: {path}'
    temp=path.with_suffix('.partial')
    with gzip.open(temp,'wt',encoding='utf8') as f: json.dump(record,f,separators=(',',':'))
    temp.rename(path)
def saved(path,frames):
    out=[]
    assert len(list(path.glob('*.json.gz')))==len(frames)
    for f in frames:
        with gzip.open(path/f'{key(f)}.json.gz','rt') as stream:r=json.load(stream)
        assert (r['sequence'],r['frame'],r['width'],r['height'])==(f.sequence,f.number,f.width,f.height)
        out.append(decode_preds(r['predictions']))
    return out
class NMSWarningGuard(logging.Handler):
    def emit(self,record):
        msg=record.getMessage()
        if 'NMS' in msg and ('time' in msg.lower() or 'limit' in msg.lower()):
            raise RuntimeError('NMS candidate retention warning: '+msg)

def inference(name,frames,directory,phase,cfg):
    directory.mkdir(parents=True,exist_ok=False)
    pred_dir=directory/'predictions';pred_dir.mkdir()
    metadata={'model':name,'phase':phase,'start':now(),'config':cfg,'config_sha256':sha(EXP/'configs/benchmark.yaml'),
              'protocol_sha256':sha(EXP/'EXPERIMENT_PROTOCOL.md'),'checkpoint_sha256':sha(ROOT/'models'/name),
              'environment_sha256':sha(EXP/'manifests/environment.json'),
              'source_manifest_sha256':sha(EXP/'manifests/source_manifest.json'),
              'frames':[key(f) for f in frames],'gpu_start':gpu(),'command':sys.argv,'resumed':False}
    write(directory/'metadata.json',metadata)
    rows=[];adapter=None;current=None;wall=time.perf_counter()
    try:
        clean();t=time.perf_counter();adapter=Adapter(ROOT/'models'/name,cfg,directory/'framework');adapter.sync()
        metadata['load_seconds']=time.perf_counter()-t
        metadata['effective_predictor_args']=vars(adapter.predictor.args)
        metadata['runtime_parameters']=adapter.runtime_parameters
        metadata['end2end']=adapter.predictor.model.end2end
        metadata['precision']=adapter.precision
        adapter.reset_peak_memory()
        for i,f in enumerate(frames):
            current=key(f);im=cv2.imread(str(f.image));assert im is not None
            ps,timing=adapter.predict(im,f.image)
            assert (timing['input_height'],timing['input_width'])==(640,640)
            t=time.perf_counter();compact=[p.compact() for p in ps];prep=(time.perf_counter()-t)*1000
            counts=adapter.predictor.post_nms_candidates
            rec={'sequence':f.sequence,'frame':f.number,'model':name,'width':f.width,'height':f.height,
                 'input_shape':[1,3,640,640],'post_nms_candidates':counts,'predictions':encode_preds(compact)}
            save_prediction(pred_dir/f'{key(f)}.json.gz',rec)
            rows.append({'sequence':f.sequence,'frame':f.number,'gt_persons':len(f.persons),'ignore_regions':f.ignore_count,
                         'post_nms_candidates':counts,'saved_predictions':len(compact),'rle_preparation_ms':prep,**timing})
            del ps,compact,im
            if (i+1)%10==0 or i==len(frames)-1:
                print(f'{phase} {name}: {i+1}/{len(frames)} {key(f)} candidates={counts}',flush=True)
        metadata.update(status='PASS',images_successful=len(rows),images_failed=0,peak_memory=adapter.peak_memory())
        csvwrite(directory/'frame_audit.csv',rows)
    except BaseException as e:
        metadata.update(status='FAIL',frame=current,exception=repr(e),traceback=traceback.format_exc(),
                        oom=isinstance(e,torch.cuda.OutOfMemoryError),images_successful=len(rows),images_failed=1,
                        gpu_failure=gpu(),memory_allocated=torch.cuda.memory_allocated(),memory_reserved=torch.cuda.memory_reserved())
        csvwrite(directory/'frame_audit.csv',rows)
        raise
    finally:
        metadata.update(end=now(),elapsed_wall_seconds=time.perf_counter()-wall,gpu_end=gpu())
        write(directory/'metadata.json',metadata)
        del adapter;clean()
    return pred_dir

def preflight(run):
    cfg=config();assert cfg['ap_max_dets']==200
    frames=frames_for(read(EXP/'manifests/preflight_frames.json'))
    base=EXP/'predictions'/run/'preflight';base.mkdir(parents=True,exist_ok=False)
    # Unmodified pilot test suite imports original files read-only; assert snapshot equality first.
    for x in read(EXP/'manifests/source_manifest.json').values():assert sha(ROOT/x['original'])==x['sha256']==sha(ROOT/x['snapshot'])
    test=cmd(str(ROOT/'.venv/bin/python'),'-B',str(ROOT/'Person_Segmentation_Pilot/src/test_pipeline.py'))
    write(EXP/'logs'/run/'evaluator_regression.json',test)
    assert test['returncode']==0,'Pilot regression failed'
    rows=[]
    for name in MODELS:
        path=base/name.removesuffix('.pt')
        log=EXP/'logs'/run/f'preflight-{name}.log'
        with log.open('w') as f,redirect_stdout(f),redirect_stderr(f):
            pred_dir=inference(name,frames,path,'preflight',cfg)
        ps=saved(pred_dir,frames)
        for cap in cfg['ap_max_dets_candidates']:
            ap,detail=ap_metrics(frames,ps,max_dets=cap)
            assert all(np.isfinite(ap[k]) for k in ['ap50','ap75','map50_95'])
            rows.append({'model':name,'max_dets':cap,**ap})
            (EXP/'logs'/run/f'ap-{name}-{cap}.txt').write_text(detail)
        del ps;gc.collect()
        print('PREFLIGHT complete',name,flush=True)
    selected=None
    for row in rows:
        ref=next(x for x in rows if x['model']==row['model'] and x['max_dets']==1000)
        for k in ['ap50','ap75','map50_95']:row[k+'_abs_difference_from_1000']=abs(row[k]-ref[k])
        row['converged']=all(row[k+'_abs_difference_from_1000']<.0001 for k in ['ap50','ap75','map50_95'])
    for cap in cfg['ap_max_dets_candidates']:
        if all(x['converged'] for x in rows if x['max_dets']==cap):selected=cap;break
    assert selected is not None
    csvwrite(EXP/'metrics'/run/'preflight_maxdet.csv',rows)
    assert all(x['converged'] for x in rows if x['max_dets']==200), 'STOP: common maxDet 200 insufficient; inspect preflight_maxdet.csv'
    selected=200
    write(EXP/'manifests'/f'{run}_preflight.json',{'status':'PASS','selected_max_dets':selected,'rows':rows,'timestamp':now()})
    assert cfg['ap_max_dets']==selected
    with (EXP/'EXPERIMENT_PROTOCOL.md').open('a') as f:
        f.write(f'\n## Preflight decision — {now()}\n\nRun `{run}` passed all three models, 100 frames each. Common AP maxDet = **{selected}**. '
                f'All three AP absolute differences from 1000 are <0.0001 for every model. '
                f'See `metrics/{run}/preflight_maxdet.csv`. This decision precedes full accuracy.\n')
    print('PHASE 3 PASS; frozen AP maxDet',selected,flush=True)

def evaluate_model(name,frames,path,out,cfg,visualize=True):
    ps=saved(path,frames);rows=[];matched=[];fixed_all=[]
    viskeys={f"{r['sequence']}_{r['frame']:06d}" for r in read(EXP/'manifests/visualization_frames.json')}
    for f,p in zip(frames,ps):
        fm=fixed_metrics(f,p,cfg['fixed_confidence'],cfg['matching_iou'],cfg['ignore_prediction_ioa'])
        fixed_all.append(fm)
        rows.append({'model':name,'sequence':f.sequence,'frame':f.number,'gt_persons':len(f.persons),'ignore_regions':f.ignore_count,
                     'predictions_ap_floor':len(p),**{k:v for k,v in fm.items() if not isinstance(v,list)}})
        matched.extend({'model':name,'sequence':f.sequence,'frame':f.number,**m} for m in fm['matches'])
        if visualize and key(f) in viskeys:
            from visualize import save_debug
            target=EXP/'outputs/visualizations'/out.name/name.removesuffix('.pt');target.mkdir(parents=True,exist_ok=True)
            save_debug(f,cv2.imread(str(f.image)),p,fm,target/f'{key(f)}.jpg',cfg['fixed_confidence'])
    aggregates=[]
    for seq in [*SEQS,'ALL']:
        ids=[i for i,f in enumerate(frames) if seq=='ALL' or f.sequence==seq]
        rr=[rows[i] for i in ids]; mm=[m for m in matched if seq=='ALL' or m['sequence']==seq]
        ap,detail=ap_metrics([frames[i] for i in ids],[ps[i] for i in ids],max_dets=cfg['ap_max_dets'])
        counts={k:sum(r[k] for r in rr) for k in ['tp','fp','fn','gt_persons','predictions_ap_floor','predictions_at_confidence','ignored_predictions']}
        a={'model':name,'sequence':seq,'frames':len(ids),**counts,**ratios(counts['tp'],counts['fp'],counts['fn']),**ap}
        for field in ['iou','dice']:
            a.update({f'matched_{field}_{k}':v for k,v in stats([m[field] for m in mm]).items()})
        aggregates.append(a)
        (out/f'{name}-{seq}-ap.txt').write_text(detail)
    csvwrite(out/'per_frame'/f'{name}.csv',rows)
    csvwrite(out/'per_instance'/f'{name}.csv',matched)
    csvwrite(out/f'{name}-aggregates.csv',aggregates)
    crowd=[]
    for n in sorted({r['gt_persons'] for r in rows}):
        rr=[r for r in rows if r['gt_persons']==n];cnt={k:sum(r[k] for r in rr) for k in ['tp','fp','fn','ignored_predictions']}
        crowd.append({'model':name,'gt_persons_per_frame':n,'frames':len(rr),**cnt,**ratios(cnt['tp'],cnt['fp'],cnt['fn'])})
    csvwrite(out/f'{name}-crowd.csv',crowd)
    del ps;gc.collect()
    return aggregates

def full(run):
    cfg=config();assert read(EXP/'manifests'/f'{run}_preflight.json')['status']=='PASS' and cfg['ap_max_dets'] is not None
    fs=frames_for();base=EXP/'predictions'/run/'accuracy';base.mkdir(parents=True,exist_ok=False)
    out=EXP/'metrics'/run;out.mkdir(parents=True,exist_ok=True)
    order=list(MODELS);random.Random(cfg['seed']).shuffle(order)
    timing_order=[order[r:]+order[:r] for r in range(3)]
    frozen={'timestamp':now(),'config_sha256':sha(EXP/'configs/benchmark.yaml'),'protocol_sha256':sha(EXP/'EXPERIMENT_PROTOCOL.md'),
            'sources':{str(p.relative_to(EXP)):sha(p) for p in (EXP/'src').rglob('*.py')},'timing_order':timing_order,'seed':cfg['seed'],
            'frame_list_sha256':sha(EXP/'manifests/images.json'),
            'sample_hashes':{n:sha(EXP/'manifests'/n) for n in ['preflight_frames.json','timing_frames.json','visualization_frames.json']}}
    frozen['shared_reference']=read(EXP/'manifests/shared_input_audit.json')
    frozen['checkpoint_manifest_sha256']=sha(EXP/'manifests/checkpoint_manifest.json')
    frozen['dataset_manifest_sha256']=sha(EXP/'manifests/dataset_manifest.json')
    write(EXP/'manifests'/f'{run}_full_freeze.json',frozen)
    archive=EXP/'reports'/run/'frozen_inputs';archive.mkdir(parents=True,exist_ok=False)
    inputs=[EXP/'EXPERIMENT_PROTOCOL.md',EXP/'configs/benchmark.yaml',*list((EXP/'src').rglob('*.py')),*list((EXP/'manifests').glob('*.json'))]
    for src in inputs:
        dst=archive/src.relative_to(EXP);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    write(EXP/'manifests'/f'{run}_input_archive.json',{'hashes':{str(src.relative_to(EXP)):sha(src) for src in inputs}})
    all_rows=[]
    for name in MODELS:
        log=EXP/'logs'/run/f'accuracy-{name}.log'
        with log.open('w') as f,redirect_stdout(f),redirect_stderr(f):
            pred_dir=inference(name,fs,base/name.removesuffix('.pt'),'accuracy',cfg)
            all_rows.extend(evaluate_model(name,fs,pred_dir,out,cfg,visualize=False))
        csvwrite(out/'per_sequence.csv',[x for x in all_rows if x['sequence']!='ALL'])
        csvwrite(out/'per_model.csv',[x for x in all_rows if x['sequence']=='ALL'])
        print('FULL ACCURACY complete',name,flush=True)
    print('PHASE 4 PASS',flush=True)

def timing(run):
    cfg=config();frozen=read(EXP/'manifests'/f'{run}_full_freeze.json')
    assert sha(EXP/'configs/benchmark.yaml')==frozen['config_sha256']
    fs=frames_for(read(EXP/'manifests/timing_frames.json'))
    base=EXP/'timing'/run;base.mkdir(parents=True,exist_ok=False)
    allrows=[];runs=[]
    for rnd,order in enumerate(frozen['timing_order'],1):
        for name in order:
            clean();before=gpu();runmeta={'round':rnd,'model':name,'start':now(),'gpu_before':before,'contaminated':contaminated(before,idle=True)}
            adapter=None;rows=[];current=None
            try:
                t=time.perf_counter();adapter=Adapter(ROOT/'models'/name,cfg,base/f'round{rnd}-{name}'/'framework');adapter.sync()
                runmeta['load_seconds']=time.perf_counter()-t
                for f in fs[:cfg['warmup_iterations']]:adapter.predict(cv2.imread(str(f.image)),f.image)
                adapter.sync();adapter.reset_peak_memory()
                runmeta['baseline_allocated_mib']=torch.cuda.memory_allocated()/1024**2
                runmeta['baseline_reserved_mib']=torch.cuda.memory_reserved()/1024**2
                runmeta['gpu_samples']=[]
                for i,f in enumerate(fs):
                    current=key(f)
                    if i%10==0:
                        g=gpu();runmeta['gpu_samples'].append(g);runmeta['contaminated']|=contaminated(g)
                    im=cv2.imread(str(f.image));ps,lat=adapter.predict(im,f.image)
                    t=time.perf_counter();comp=[p.compact() for p in ps];prep=(time.perf_counter()-t)*1000
                    rows.append({'model':name,'round':rnd,'sequence':f.sequence,'frame':f.number,
                                 'post_nms_candidates':adapter.predictor.post_nms_candidates,'predictions':len(ps),'rle_preparation_ms':prep,**lat})
                    del ps,comp,im
                runmeta.update(adapter.peak_memory());runmeta['status']='PASS'
            except BaseException as e:
                runmeta.update(status='FAIL',frame=current,exception=repr(e),traceback=traceback.format_exc(),gpu_failure=gpu())
                raise
            finally:
                runmeta['end']=now();runmeta['gpu_after']=gpu();runmeta['contaminated']|=contaminated(runmeta['gpu_after'])
                write(base/f'round{rnd}-{name}.json',runmeta)
                for r in rows:r['contaminated']=runmeta['contaminated']
                csvwrite(base/f'round{rnd}-{name}.csv',rows)
                del adapter;clean()
            runs.append(runmeta);allrows.extend(rows)
            print('TIMING',rnd,name,'CONTAMINATED' if runmeta['contaminated'] else 'CLEAN',flush=True)
    csvwrite(base/'all_frames.csv',allrows)
    summary=[]
    for name in MODELS:
        rr=[r for r in allrows if r['model']==name and not r['contaminated']]
        good=[r for r in runs if r['model']==name and not r['contaminated']]
        row={'model':name,'clean_rounds':len(good),'frames':len(rr)}
        for metric in ['preprocess_ms','inference_ms','postprocess_ms','ultralytics_postprocess_inclusive_ms','total_ms','rle_preparation_ms']:
            row.update({metric+'_'+k:v for k,v in stats([r[metric] for r in rr]).items()})
        row['fps']=1000/row['total_ms_mean'] if rr else None
        for k in ['peak_gpu_allocated_mib','peak_gpu_reserved_mib','baseline_allocated_mib','baseline_reserved_mib']:
            row[k]=max(r[k] for r in good) if good else None
        row['load_seconds_mean']=float(np.mean([r['load_seconds'] for r in good])) if good else None
        summary.append(row)
    csvwrite(base/'summary.csv',summary)
    write(base/'runs.json',runs)
    print('PHASE 5', 'PASS' if all(r['clean_rounds']==3 for r in summary) else 'INCOMPLETE: contaminated timing',flush=True)

def main():
    phase=sys.argv[1];run=sys.argv[2]
    torch.manual_seed(20260929);np.random.seed(20260929);random.seed(20260929)
    torch.backends.cudnn.benchmark=False
    from ultralytics.utils import LOGGER
    LOGGER.addHandler(NMSWarningGuard())
    (EXP/'logs'/run).mkdir(parents=True,exist_ok=True)
    try:
        {'preflight':preflight,'full':full,'timing':timing}[phase](run)
    except BaseException as e:
        write(EXP/'logs'/run/f'{phase}_FAILURE.json',{'phase':phase,'timestamp':now(),'exception':repr(e),'traceback':traceback.format_exc(),'gpu':gpu()})
        traceback.print_exc();print('BENCHMARK STOP: '+phase+' failed',flush=True);sys.exit(1)
if __name__=='__main__':main()
