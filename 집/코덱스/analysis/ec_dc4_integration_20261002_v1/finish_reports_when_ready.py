"""실행 중인 고정 큐 완료 시 검산된 연구 보고서 자동 생성. 모델 추가 실행 없음."""
from pathlib import Path
import sys,time,subprocess,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
def main():
    deadline=time.monotonic()+48*3600
    while not (OUT/'queue_complete.json').exists():
        assert time.monotonic()<deadline,'48시간 내 큐 완료 표식 없음. 담당자가 큐 오류/프로세스를 확인해야 함.'
        time.sleep(10)
    record=json.loads((OUT/'queue_complete.json').read_text(encoding='utf-8'))
    assert record['status']=='COMPLETE'
    subprocess.run([sys.executable,'-B','-u',str(HERE/'build_reports.py')],check=True)
    with (OUT/'reports_complete.json').open('x',encoding='utf-8') as stream:
        json.dump({'status':'COMPLETE','reports':[str(HERE/'결과보고서_v1.md'),str(HERE.parent/'ec_stage2_tabpfn_20261002_v2/결과보고서_v1.md')],
                   'platform_submission':False,'shared_handoff_update_pending':True},stream,ensure_ascii=False,indent=2)
    print('두 연구 보고서 및 검산 완료 기록 생성. 공용 인계 갱신 필요.',flush=True)
if __name__=='__main__':main()
