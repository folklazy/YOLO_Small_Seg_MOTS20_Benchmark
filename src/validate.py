"""Phase 1: read-only inputs, official checkpoint retrieval, frozen manifests."""
from pathlib import Path
import os, sys, json, hashlib, platform, subprocess, shutil, datetime, configparser, urllib.request
ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(EXP/'logs/ultralytics_config')
os.environ['YOLO_AUTOINSTALL'] = 'false'
sys.dont_write_bytecode = True
SEQS = ['MOTS20-02','MOTS20-05','MOTS20-09','MOTS20-11']
MODELS = ['yolo26s-seg.pt','yolo11s-seg.pt','yolov8s-seg.pt']
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def cmd(*a):
    p=subprocess.run(a,cwd=ROOT,text=True,capture_output=True)
    return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def read_json(path): return json.loads(path.read_text())
def main():
    for d in ['configs','manifests','metrics/per_frame','metrics/per_instance','predictions','timing','logs','reports','outputs/plots','outputs/visualizations','src/frozen_pilot']:
        (EXP/d).mkdir(parents=True,exist_ok=True)
    import importlib.metadata as md
    import torch, torchvision, ultralytics, numpy as np
    from pycocotools import mask as cm
    env={'timestamp':now(),'hostname':platform.node(),'os':platform.platform(),'python':sys.version,'cwd':str(ROOT),
         'packages':{p:md.version(p) for p in ['ultralytics','torch','torchvision','numpy','pycocotools','Pillow','opencv-python','matplotlib']},
         'torch_version':torch.__version__,'torchvision_version':torchvision.__version__,'cuda_runtime':torch.version.cuda,
         'cuda_available':torch.cuda.is_available(),'CUDA_VISIBLE_DEVICES':os.getenv('CUDA_VISIBLE_DEVICES'),
         'gpu':str(torch.cuda.get_device_properties(0)) if torch.cuda.is_available() else None,
         'nvidia_smi':cmd('nvidia-smi'),'gpu_details':cmd('nvidia-smi','-q'),
         'git_commit':cmd('git','rev-parse','HEAD'),'git_status':cmd('git','status','--short'),'git_diff':cmd('git','diff'),
         'freeze':cmd(str(ROOT/'.venv/bin/python'),'-m','pip','freeze')}
    write(EXP/'manifests/environment.json',env)
    assert torch.cuda.is_available(), 'CUDA required'
    sources={}
    for name in ['mots.py','metrics.py','yolo_adapter.py','visualize.py','test_pipeline.py']:
        src=ROOT/'YOLO_Large_Seg_MOTS20_Benchmark/src/frozen_pilot'/name; dst=EXP/'src/frozen_pilot'/name
        if dst.exists(): assert sha(src)==sha(dst)
        else: shutil.copyfile(src,dst)
        sources[name]={'original':str(src.relative_to(ROOT)),'snapshot':str(dst.relative_to(ROOT)),'sha256':sha(src)}
    write(EXP/'manifests/source_manifest.json',sources)
    sys.path.insert(0,str(EXP/'src/frozen_pilot'))
    from mots import load_frames, validate_known_gt
    dataset=ROOT/'datasets/MOTS/MOTS/train'
    assert sorted(p.name for p in dataset.iterdir() if p.is_dir())==SEQS
    all_images=[]; sequences=[]; samples=[]; vis=[]; instances=[]; crowd=[]
    for seq in SEQS:
        base=dataset/seq; c=configparser.ConfigParser(); c.read(base/'seqinfo.ini'); meta=c['Sequence']; n=int(meta['seqLength'])
        names=[f'{i:06d}{meta["imExt"]}' for i in range(1,n+1)]
        assert sorted(p.name for p in (base/'img1').iterdir() if p.is_file())==names
        # Validate every GT line, including annotations that a frame loader might otherwise skip.
        classes={}
        for line in (base/'gt/gt.txt').read_text().splitlines():
            f,oid,cls,h,w,rle=line.split(); assert 1<=int(f)<=n and int(cls) in (2,10)
            classes[cls]=classes.get(cls,0)+1
        frames=load_frames(dataset,seq,list(range(1,n+1)))
        if seq==SEQS[0]: validate_known_gt(frames[0])
        for f in frames:
            all_images.append({'sequence':seq,'frame':f.number,'path':str(f.image.relative_to(ROOT)),
                               'width':f.width,'height':f.height,'sha256':sha(f.image),'bytes':f.image.stat().st_size})
            crowd.append({'sequence':seq,'frame':f.number,'gt_persons':len(f.persons),'ignore_regions':f.ignore_count})
            for g in f.persons:
                a=int(cm.area(g.rle)); x,y,bw,bh=cm.toBbox(g.rle).tolist()
                instances.append({'sequence':seq,'frame':f.number,'object_id':g.object_id,'mask_area':a,'bbox_width':bw,'bbox_height':bh,
                                  'bbox_area':bw*bh,'image_width':f.width,'image_height':f.height,'relative_mask_area':a/(f.width*f.height),
                                  'relative_bbox_area':bw*bh/(f.width*f.height)})
        sequences.append({'sequence':seq,'frames':n,'resolution_wh':[frames[0].width,frames[0].height],
                          'gt_sha256':sha(base/'gt/gt.txt'),'seqinfo_sha256':sha(base/'seqinfo.ini'),'classes':classes})
        samples.extend({'sequence':seq,'frame':int(i)} for i in np.linspace(1,n,25).round().astype(int))
        vis.extend({'sequence':seq,'frame':int(i)} for i in np.linspace(1,n,3).round().astype(int))
        print('Validated',seq,n,'frames',flush=True)
    assert len(all_images)==2862
    write(EXP/'manifests/images.json',all_images)
    write(EXP/'manifests/preflight_frames.json',samples)
    write(EXP/'manifests/timing_frames.json',samples)
    write(EXP/'manifests/visualization_frames.json',vis)
    write(EXP/'manifests/dataset_manifest.json',{'dataset_root':str(dataset.relative_to(ROOT)),'sequences':sequences,'total_frames':len(all_images),
          'image_manifest':'manifests/images.json','image_manifest_sha256':sha(EXP/'manifests/images.json'),
          'validation_timestamp':now(),'checks':'all images decoded; all Person and ignore RLE decoded; dimensions, classes, IDs and frame continuity checked'})
    import csv
    for name,rows in [('gt_instances',instances),('gt_frames',crowd)]:
        with (EXP/f'metrics/{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    write(EXP/'metrics/person_size_distribution.json',dict(zip(['min','p10','p25','median','p75','p90','max'],np.percentile([x['relative_mask_area'] for x in instances],[0,10,25,50,75,90,100]).tolist())))
    previous=ROOT/'YOLO_Large_Seg_MOTS20_Benchmark'
    for filename in ['images.json','preflight_frames.json','timing_frames.json','visualization_frames.json']:
        assert read_json(EXP/'manifests'/filename)==read_json(previous/'manifests'/filename), 'STOP: dataset/sample mismatch'
        shutil.copyfile(previous/'manifests'/filename,EXP/'manifests'/filename)
    assert sequences==read_json(previous/'manifests/dataset_manifest.json')['sequences'], 'STOP: GT/metadata mismatch'
    for filename in ['gt_instances.csv','gt_frames.csv','person_size_distribution.json']:
        assert sha(EXP/'metrics'/filename)==sha(previous/'metrics'/filename), 'STOP: GT descriptive data mismatch'
    previous_env=read_json(previous/'manifests/environment.json')
    comparison={k:{'previous':previous_env[k],'current':env[k]} for k in ['hostname','os','python','packages','torch_version','torchvision_version','cuda_runtime','cuda_available','CUDA_VISIBLE_DEVICES','gpu'] if previous_env[k]!=env[k]}
    write(EXP/'manifests/environment_comparison.json',{'differences':comparison})
    assert not comparison, 'STOP: environment mismatch; report before proceeding'
    records=[]
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    for name in MODELS:
        p=ROOT/'models'/name; url=f'https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}'
        if not p.exists():
            print('Downloading official checkpoint',url,flush=True)
            part=p.with_suffix('.pt.partial')
            urllib.request.urlretrieve(url,part)
            part.rename(p)
        model=YOLO(str(p),task='segment')
        assert model.task=='segment' and model.names[0]=='person', 'Unexpected task / Person mapping: STOP'
        params=sum(x.numel() for x in model.model.parameters())
        model.model.end2end=False
        info=model.info(verbose=True,imgsz=640)
        flops=get_flops(model.model,imgsz=640)
        records.append({'family':name.split('-')[0],'filename':name,'source':url,'bytes':p.stat().st_size,'sha256':sha(p),
                        'task':model.task,'names':model.names,'parameters_loaded':params,'gflops_nms_unfused':flops if flops>0 else None,
                        'ultralytics_info':info,'end2end_for_profiling':model.model.end2end})
        write(EXP/'manifests/checkpoint_manifest.json',records)
        print('Validated checkpoint',name,params,'parameters',flush=True)
        del model
    print('PHASE 1 PASS',flush=True)
if __name__=='__main__': main()
