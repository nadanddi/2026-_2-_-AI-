"""사용자 직접 전달 추가 요청: DC4 단회 잠금 확인 먼저, 이후 K2/K3."""
from pathlib import Path
import os,sys,json,subprocess,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
LOCAL=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
K2=HERE.parent/'ec_stage2_tabpfn_20261002_v2'
os.environ['PYTHONPATH']='';os.environ['PYTHONIOENCODING']='utf-8'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='4'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def call(script,*args):
    print(f'실행: {script.name} {" ".join(args)}',flush=True)
    subprocess.run([sys.executable,'-B','-u',str(script),*args],check=True,env=os.environ.copy())

def main():
    assert all((LOCAL/f'shard{i}_r3_complete.json').exists() for i in range(3))
    if not (HERE/'et_replication_result.json').exists():call(HERE/'run_dc4.py','--collect','et')
    else:
        audit=read(HERE/'et_reproduction_quick_audit_v1.json')
        assert audit['status']=='PASS' and audit['OOF_sha256']==sha(LOCAL/'et_replication_oof.csv')
        assert read(HERE/'et_replication_result.json')['replication_pass']
        print('완료 ET 독립 재현 산출물 검산 기록 및 해시 확인',flush=True)
    for i in range(3):
        if not (LOCAL/f'shard{i}_pfn_complete.json').exists():call(HERE/'run_dc4.py','--component','pfn','--shard',str(i))
    if not (HERE/'v2_integration_result.json').exists():call(HERE/'run_dc4.py','--collect','v2')
    for tag,arm in [('et_replication','season_et'),('v2_integration','season_v2')]:
        path=HERE/f'{tag}_independent_verification.json'
        if path.exists():
            report=read(path);assert report['status']=='PASS' and report['csv_sha256']==sha(LOCAL/f'{tag}_oof.csv')
        else:
            code=f"import sys;sys.path.insert(0,{str(HERE)!r});import verify_public;verify_public.verify({tag!r},{arm!r})"
            subprocess.run([sys.executable,'-B','-u','-c',code],check=True,env=os.environ.copy())
    ledger=LOCAL/'locked_confirmation/LOCK_CONSUMED_ONCE.json'
    if ledger.exists():
        assert (HERE/'locked_confirmation_result.json').exists(),'잠금 소비 후 결과 미완료. 재열람 금지, 수동 감사 필요.'
        assert read(HERE/'locked_confirmation_result.json')['status']=='CONSUMED'
    else:call(HERE/'score_lock_once.py')
    print('DC4 공개 검증 및 잠금 단회 확인 완료. 기존 K2/K3 요청 계산으로 이동.',flush=True)
    jobs=[subprocess.Popen([sys.executable,'-B','-u',str(K2/'run_stage2_tabpfn.py'),'--run','--component','r3','--shard',str(i)],env=os.environ.copy()) for i in range(3)]
    codes=[job.wait() for job in jobs];assert codes==[0,0,0],f'K2 R3 워커 실패: {codes}'
    for i in range(3):call(K2/'run_stage2_tabpfn.py','--run','--component','pfn','--shard',str(i))
    call(K2/'run_stage2_tabpfn.py','--collect')
    if not (K2/'independent_verification_v2.json').exists():call(K2/'verify_stage2_v2.py')
    assert read(K2/'independent_verification_v2.json')['status']=='PASS'
    record={'status':'COMPLETE','order_version':3,'stages':['DC4_ET','DC4_v2','locked_once','K2_K3'],
            'platform_submission':False,'candidate_adopted':False,'final_review_pending':True}
    with (LOCAL/'queue_complete.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
    print('고정 연구 계산 완료. 결과 보고서 자동 생성기가 이어서 실행합니다.',flush=True)

if __name__=='__main__':main()
