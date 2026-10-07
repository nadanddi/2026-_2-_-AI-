"""Run one registered BLK diagnostic stage after current-source checks.

All commands are fixed argv, no shell. Does not replace each stage's own gate.
Preserves old outputs and writes one new execution receipt per completed stage.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
STAGES={'receipt':'blk_cached_pfn_receipt_v2.py','assemble':'blk_assemble_predictions_v4.py',
    'verify':'verify_BLK_baseline_v4.py','score':'blk_score_v6.py'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=list(STAGES),required=True)
    stage=parser.parse_args().stage
    rp=HERE/'BLK_execution_registration_v1.json'
    reg=json.loads(rp.read_text(encoding='utf-8-sig'))
    assert reg['status']=='REGISTERED_BEFORE_FIRST_HELDOUT_SCORE' and reg['stages']==STAGES
    assert reg['executor_sha256']==sha(__file__)
    assert reg['screened_variant_count']==6 and reg['adoption_permitted'] is False
    pins=reg['frozen_sources_sha256']; assert len(pins)>=40
    assert all(Path(p).is_absolute() and sha(p)==s for p,s in pins.items())
    program=HERE/STAGES[stage]; assert pins[str(program.resolve())]==sha(program)
    out=HERE/f'BLK_execution_{stage}_v1.json'; assert not out.exists(), 'Preserve previous stage receipt'
    environment=dict(os.environ,PYTHONPATH='',PYTHONIOENCODING='utf-8')
    started=time.monotonic()
    result=subprocess.run([sys.executable,'-u',str(program)],cwd=str(ROOT),env=environment,check=False)
    after={p:sha(p) for p in pins}; assert after==pins, 'Registered source changed during stage'
    receipt={'stage':stage,'program':str(program),'returncode':result.returncode,'registration_sha256':sha(rp),
        'executor_sha256':sha(__file__),'duration_seconds':time.monotonic()-started,
        'frozen_sources_unchanged':True,'heldout_score_stage':stage=='score','adoption_permitted':False,
        'scope':'Execution/source receipt only; the selected stage owns its numeric and label-boundary gates'}
    temporary=out.with_suffix('.tmp')
    with temporary.open('x',encoding='utf-8') as f: json.dump(receipt,f,ensure_ascii=False,indent=2)
    os.replace(temporary,out)
    if result.returncode: raise SystemExit(result.returncode)
    print(f'Registered {stage} stage completed; sources unchanged',flush=True)

if __name__=='__main__': main()
