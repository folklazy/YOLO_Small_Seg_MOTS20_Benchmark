"""Small timing: frozen measured loop, idle gate and retry contaminated rounds only."""
import inspect
import shutil
import time
import benchmark as b

def wait_idle():
    time.sleep(2)
    while b.contaminated(b.gpu(),idle=True):
        time.sleep(10)

def main(run):
    source=inspect.getsource(b.timing)
    source=source.replace('def timing(run):','def clean_timing(run):')
    old="base=EXP/'timing'/run;base.mkdir(parents=True,exist_ok=False)"
    assert source.count(old)==1
    source=source.replace(old,"base=EXP/'timing'/run/'clean_repetition';base.mkdir(parents=True,exist_ok=False)")
    start=source.index('            clean();before=gpu();runmeta=')
    end=source.index('            runs.append(runmeta);allrows.extend(rows)',start)
    original=source[start:end]
    # The inner measured loop, warmups, telemetry and statistics remain verbatim.
    body=original.replace('clean();before=gpu();runmeta=','clean();wait_idle();before=gpu();runmeta=')
    body=body.replace("base/f'round", "attempt_dir/f'round")
    body='\n'.join('    '+line if line else line for line in body.splitlines())+'\n'
    replacement='''            attempt=0
            while True:
                attempt_dir=base/'attempts'/f'round{rnd}-{name}-attempt{attempt}'
                attempt_dir.mkdir(parents=True,exist_ok=False)
'''+body+'''                if not runmeta['contaminated']:
                    for suffix in ['json','csv']:
                        target=base/f'round{rnd}-{name}.{suffix}'
                        assert not target.exists()
                        shutil.copyfile(attempt_dir/f'round{rnd}-{name}.{suffix}',target)
                    break
                attempt+=1
                assert attempt<10,'Repeated GPU contamination; preserved attempts, stop'
'''
    source=source[:start]+replacement+source[end:]
    path=b.EXP/'logs'/run/'clean_timing_source.py'
    assert not path.exists();path.write_text(source)
    b.write(b.EXP/'logs'/run/'clean_timing_provenance.json',{
        'timestamp':b.now(),'original_runner_sha256':b.sha(b.EXP/'src/benchmark.py'),
        'generated_source_sha256':b.sha(path),'measured_loop_unchanged':True,
        'changes':['idle gate before load outside measured loop','fresh attempt directories',
                   'exclude/preserve contaminated attempts; repeat that round/model only'],
        'primary_timing':f'timing/{run}/clean_repetition'})
    namespace=dict(vars(b));namespace.update(wait_idle=wait_idle,shutil=shutil)
    exec(compile(source,str(path),'exec'),namespace)
    namespace['clean_timing'](run)
