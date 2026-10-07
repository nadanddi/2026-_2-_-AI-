"""Frozen stage executor; gate dependencies and before/after source checks."""
from pathlib import Path
import json,hashlib,subprocess,sys,os,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def main():
    assert len(sys.argv)==3 and sys.argv[1]=='--stage'
    stage=sys.argv[2]
    regpath=HERE/'DOMAIN24_BLK_pipeline_registration_v2.json'
    reg=read(regpath)
    assert reg['status']=='REGISTERED_BEFORE_DOMAIN_SCORE'
    assert reg['executor_sha256']==sha(__file__)
    stages={'raw_receipt':'domain_raw_receipt_v1.py','assemble':'assemble_domain_BLK_v2.py',
            'verify':'verify_domain_BLK_v1.py','score':'score_domain_BLK_v1.py'}
    assert reg['stages']==stages and stage in stages
    assert all(sha(path)==value for path,value in reg['sources_sha256'].items())
    output=HERE/f'DOMAIN24_execution_{stage}_v1.json';assert not output.exists()
    if stage=='raw_receipt':
        complete=read(HERE/'checkpoints/DOMAIN24_BLK_RAW_v3/complete.json')
        assert complete['status']=='DOMAIN24_RAW_ET_COMPLETE_NOT_SCORED'
    else:
        prior={'assemble':'raw_receipt','verify':'assemble','score':'verify'}[stage]
        previous=read(HERE/f'DOMAIN24_execution_{prior}_v1.json')
        assert previous['returncode']==0 and previous['frozen_sources_unchanged'] is True
        assert previous['pipeline_registration_sha256']==sha(regpath)
    environment=dict(os.environ,PYTHONPATH='',PYTHONIOENCODING='utf-8')
    started=time.monotonic()
    process=subprocess.run([sys.executable,'-u',str(HERE/stages[stage])],cwd=ROOT,env=environment)
    unchanged=all(sha(path)==value for path,value in reg['sources_sha256'].items())
    record={'stage':stage,'returncode':process.returncode,'pipeline_registration_sha256':sha(regpath),
            'executor_sha256':sha(__file__),'frozen_sources_unchanged':unchanged,
            'duration_seconds':time.monotonic()-started,'heldout_score_stage':stage=='score','adoption_permitted':False}
    temporary=output.with_suffix('.tmp')
    with temporary.open('x',encoding='utf-8') as handle:json.dump(record,handle,ensure_ascii=False)
    os.replace(temporary,output)
    assert process.returncode==0 and unchanged
    print(f'Registered domain {stage} stage complete, sources unchanged',flush=True)
if __name__=='__main__':main()
