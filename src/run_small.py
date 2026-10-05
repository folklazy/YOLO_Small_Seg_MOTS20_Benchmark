"""Deterministic, singleton Small-only runner. No Nano execution or mid-run reporting."""
from pathlib import Path
import argparse
import contextlib
import csv
import fcntl
import gzip
import json
import math
import os
import random
import shutil
import sys
import traceback
from validate import ROOT,EXP,MODELS,sha,write,now

EXPECTED=['yolo26s-seg.pt','yolo11s-seg.pt','yolov8s-seg.pt']
def read(p):return json.loads(p.read_text())

class PhaseLog:
    """Preserve verbose log while emitting model completion lines only."""
    def __init__(self,stream,terminal):self.stream=stream;self.terminal=terminal;self.buffer=''
    def write(self,s):
        self.stream.write(s);self.stream.flush();self.buffer+=s
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            if line.startswith('FULL ACCURACY complete '):
                cp=line.split()[-1];i=EXPECTED.index(cp)+1
                print(f'[{i}/3] {cp.removesuffix("-seg.pt").replace("yolo","YOLO")} accuracy: COMPLETE',file=self.terminal,flush=True)
        return len(s)
    def flush(self):self.stream.flush()

def phase(run,name,fn):
    with (EXP/'logs'/run/f'{name}.log').open('a') as log:
        sink=PhaseLog(log,sys.stdout)
        with contextlib.redirect_stdout(sink),contextlib.redirect_stderr(log):return fn()

def shared_validation(run):
    import validate
    validate.main()
    import ultralytics
    reference=ROOT/'YOLO_Large_Seg_MOTS20_Benchmark'
    framework=read(reference/'manifests/framework_source_hashes.json')
    assert all(sha(Path(ultralytics.__file__).parent/f)==h for f,h in framework.items()),'Framework drift'
    write(EXP/'manifests/framework_source_hashes.json',framework)
    for n in ['images.json','preflight_frames.json','timing_frames.json','visualization_frames.json']:
        assert sha(reference/'manifests'/n)==sha(EXP/'manifests'/n)
    assert not read(EXP/'manifests/environment_comparison.json')['differences']
    write(EXP/'manifests/shared_input_audit.json',{'timestamp':now(),'status':'PASS',
        'dataset_compatibility':'PASS','environment_compatibility':'PASS','framework_compatibility':'PASS',
        'same_frame_order_hashes_dimensions_gt_metadata':True,'reference_repository':reference.name,
        'reference_hashes':{n:sha(reference/'manifests'/n) for n in ['images.json','dataset_manifest.json',
            'preflight_frames.json','timing_frames.json','visualization_frames.json','framework_source_hashes.json']}})
    archive=EXP/'reports'/run/'preflight_inputs';archive.mkdir(parents=True,exist_ok=False)
    for p in [EXP/'EXPERIMENT_PROTOCOL.md',EXP/'configs/benchmark.yaml',*list((EXP/'src').rglob('*.py')),*list((EXP/'manifests').glob('*.json'))]:
        dst=archive/p.relative_to(EXP);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)

def check_preflight(run):
    decision=read(EXP/'manifests'/f'{run}_preflight.json')
    assert decision['status']=='PASS' and decision['selected_max_dets']==200
    import numpy as np
    from pycocotools import mask as cm
    checks=[]
    for cp in EXPECTED:
        base=EXP/'predictions'/run/'preflight'/cp.removesuffix('.pt');m=read(base/'metadata.json')
        assert m['status']=='PASS' and m['images_successful']==100 and m['images_failed']==0
        assert m['end2end'] is False and m['precision']=='fp32'
        for f in read(EXP/'manifests/preflight_frames.json'):
            with gzip.open(base/'predictions'/f"{f['sequence']}_{f['frame']:06d}.json.gz",'rt') as stream:r=json.load(stream)
            assert r['input_shape']==[1,3,640,640] and r['post_nms_candidates']<1000
            for p in r['predictions']:
                assert p['class']==0 and math.isfinite(p['confidence']) and all(math.isfinite(x) for x in p['bbox_xyxy'])
                rle={'size':p['rle']['size'],'counts':p['rle']['counts'].encode('ascii')}
                mask=cm.decode(rle);assert mask.shape==(r['height'],r['width']) and mask.any()
        checks.append({'model':cp,'frames':100,'finite_output':True,'native_nonempty_masks':True,'status':'PASS'})
    write(EXP/'manifests'/f'{run}_preflight_output_checks.json',{'status':'PASS','checks':checks})

def main(run):
    assert MODELS==EXPECTED
    phase(run,'shared_validation',lambda:shared_validation(run))
    import benchmark as b
    b.torch.manual_seed(20260929);b.np.random.seed(20260929);random.seed(20260929)
    b.torch.backends.cudnn.benchmark=False
    from ultralytics.utils import LOGGER
    LOGGER.addHandler(b.NMSWarningGuard())
    phase(run,'preflight',lambda:b.preflight(run))
    phase(run,'preflight_output_checks',lambda:check_preflight(run))
    print('[SMALL] preflight: PASS',flush=True)
    phase(run,'full_accuracy',lambda:b.full(run))
    import timing_small
    phase(run,'timing',lambda:timing_small.main(run))
    timing=list(csv.DictReader((EXP/'timing'/run/'clean_repetition/summary.csv').open()))
    assert len(timing)==3 and all(x['clean_rounds']=='3' and x['frames']=='300' for x in timing)
    for i,cp in enumerate(EXPECTED,1):print(f'[{i}/3] {cp.removesuffix("-seg.pt").replace("yolo","YOLO")} timing: COMPLETE',flush=True)
    import build_small_results
    phase(run,'build_metrics',lambda:build_small_results.main(run))
    print('[SMALL] metrics generated',flush=True)
    import report_small
    phase(run,'reports',lambda:report_small.main(run))
    assert read(EXP/'manifests/DOCUMENT_VALIDATION.json')['status']=='PASS'
    for p,h in read(EXP/'manifests/OTHER_TIERS_GUARD.json').items():assert sha(ROOT/p)==h,('Other tier changed',p)
    state_path=ROOT/'YOLO_Instance_Segmentation_MOTS20_Scaling_Study/STUDY_STATE.json'
    state=read(state_path);before=json.loads(json.dumps(state))
    state['tiers']['small'].update(completion_status='COMPLETE',result_status='PASS_WITH_WARNINGS',
        canonical_run_id=run,standardized=True,last_verified=now(),
        notes='Small only: 3 models × 2862 frames; AP maxDet200 convergence PASS; 9 clean timing rounds; scientific/document validation PASS. No Nano execution. Explicit user-requested summary layouts.')
    state['last_verified']=now()
    assert state['tiers']['nano']==before['tiers']['nano']
    write(EXP/'manifests/STUDY_STATE_UPDATE.json',{'timestamp':now(),'before':before,'after':state})
    write(state_path,state)
    write(EXP/'manifests/SMALL_COMPLETE.json',{'timestamp':now(),'status':'PASS_WITH_WARNINGS','run_id':run,
        'scientific_validation':'PASS','document_validation':'PASS','nano_started':False})
    print('[SMALL] validation: PASS',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--all',action='store_true',required=True)
    parser.add_argument('--run-id',required=True);args=parser.parse_args()
    (EXP/'logs'/args.run_id).mkdir(parents=True,exist_ok=True)
    with (EXP/'logs/SMALL_RUNNER.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(EXP/'manifests/SMALL_ACTIVE_RUN.json',{'run_id':args.run_id,'pid':os.getpid(),'started':now()})
        try:main(args.run_id)
        except BaseException as e:
            write(EXP/'logs'/args.run_id/'RUNNER_FAILURE.json',{'timestamp':now(),'error':repr(e),'traceback':traceback.format_exc()})
            print('[SMALL] STOP: '+str(e),flush=True);sys.exit(1)
