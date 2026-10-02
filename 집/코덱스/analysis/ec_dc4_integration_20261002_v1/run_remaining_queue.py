"""고정 연구 순차 실행. 큰 PFN은 동시에 하나만. 완료 산출물은 덮어쓰지 않음."""
from pathlib import Path
import os,sys,json,time,subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
LOCAL=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
K2=HERE.parent/'ec_stage2_tabpfn_20261002_v2'
os.environ['PYTHONPATH']='';os.environ['PYTHONIOENCODING']='utf-8'
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'

def call(script,*args):
    print(f'실행: {script.name} {" ".join(args)}',flush=True)
    subprocess.run([sys.executable,'-B','-u',str(script),*args],check=True,env=os.environ.copy())

def main():
    # Three R3 workers were started by the parent before this queue.
    start=time.monotonic();last=0
    while not all((LOCAL/f'shard{i}_r3_complete.json').exists() for i in range(3)):
        assert time.monotonic()-start<3600,'R3 완료 표식 미생성. 담당자가 워커 종료 원인을 확인해야 함.'
        if time.monotonic()-last>60:print('DC4 R3 3그룹 완료 대기',flush=True);last=time.monotonic()
        time.sleep(5)
    call(HERE/'run_dc4.py','--collect','et')
    for i in range(3):call(HERE/'run_dc4.py','--component','pfn','--shard',str(i))
    call(HERE/'run_dc4.py','--collect','v2')
    call(HERE/'verify_public.py')
    jobs=[subprocess.Popen([sys.executable,'-B','-u',str(K2/'run_stage2_tabpfn.py'),'--run','--component','r3','--shard',str(i)],env=os.environ.copy()) for i in range(3)]
    assert all(job.wait()==0 for job in jobs),'K2 R3 워커 실패'
    for i in range(3):call(K2/'run_stage2_tabpfn.py','--run','--component','pfn','--shard',str(i))
    call(K2/'run_stage2_tabpfn.py','--collect')
    call(K2/'verify_stage2_v1.py')
    call(HERE/'score_lock_once.py')
    record={'status':'COMPLETE','finished_time_local':time.strftime('%Y-%m-%d %H:%M:%S'),
            'stages':['DC4_ET','DC4_v2','K2_K3','public_independent_verification','locked_once'],
            'platform_submission':False,'candidate_adopted':False,'final_review_pending':True}
    with (LOCAL/'queue_complete.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
    print('고정 연구 계산 완료. 최종 해석·인계 기록은 별도 검토 필요.',flush=True)

if __name__=='__main__':main()
