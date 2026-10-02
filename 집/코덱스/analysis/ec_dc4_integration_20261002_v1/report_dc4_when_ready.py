"""DC4 단회 확인 직후 별도 판정 보고서 생성. K2 완료를 기다리지 않는다."""
from pathlib import Path
import sys,time,json,hashlib,math
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
LOCAL=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    deadline=time.monotonic()+48*3600
    target=HERE/'locked_confirmation_result.json'
    while not target.exists():
        assert time.monotonic()<deadline,'DC4 결과 대기 48시간 초과. 큐 진행 상태를 확인해야 함.'
        time.sleep(10)
    et=read(HERE/'et_replication_result.json')
    audit=read(HERE/'et_replication_independent_verification.json')
    va=read(HERE/'v2_integration_independent_verification.json')
    result=read(HERE/'v2_integration_result.json');lock=read(target)
    assert et['replication_pass'] and audit['status']==va['status']=='PASS'
    assert audit['csv_sha256']==sha(LOCAL/'et_replication_oof.csv')
    assert va['csv_sha256']==sha(LOCAL/'v2_integration_oof.csv')
    assert lock['status']=='CONSUMED' and lock['labels_read_once'] and lock['independent_rmse_pass']
    assert lock['n_labels']==960 and math.isclose(lock['delta_rmse'],lock['season_rmse']-lock['day_rmse'],abs_tol=1e-15)
    for name,digest in lock['prediction_files'].items():assert sha(LOCAL/'locked_confirmation'/name)==digest
    public=va['direction_pass'] and va['confidence_pass']
    adopted=public and lock['locked_gate_pass']
    rows=[]
    for s in va['independent_scores']:
        rows.append(f"| {s['validator']} | {s['seed']} | {s['reference_rmse']:.9f} | {s['candidate_rmse']:.9f} | {s['improved']} |")
    text=f'''# DC4 조건부 후보 판정 · 집 코덱스

판정: **{'모든 조건 통과' if adopted else '정식 후보 조건 미충족'}**. 독립 재현, v2 통합 공개 기준, 잠금 단회 확인을 모두 통과해야 한다는 사용자 결정을 그대로 적용했다. 성능표와 판정은 독립 CSV 산술·부트스트랩 검산 후 작성했다. 실제 제출은 하지 않았다.

- 독립 ET 재현: PASS. 22폴드×3시드 예측 최대차 {max(x['ET_max_gap'] for x in et['independent_reproduction']):.3g}. 원본 14/15칸 개선 및 EXT10/101 악화도 재현했다. ET 단독을 정식 후보로 채택한 것이 아니다.
- v2 공개 판정: {'PASS' if public else 'FAIL'}, {sum(x['improved'] for x in va['independent_scores'])}/15칸 개선. 모든 시드×5검증기 개선 및 DIAG10 세 검정 p<.0125/97.5% CI상한<0을 요구했다. 이번 두 연구 후보에 k2 보정. 후반 구간 수치는 진단이며 기준을 바꾸는 데 쓰지 않았다.
- 잠금 40일 단회: {'PASS' if lock['locked_gate_pass'] else 'FAIL'}. 같은7344행306일 학습, 같은R3시드7/101/2024 및 PFN문맥1..4 평균으로 예측을 먼저 저장한 뒤960행 정답을 한 번 읽었다. day v2 RMSE {lock['day_rmse']:.9f}, 계절 v2 RMSE {lock['season_rmse']:.9f}, 차 {lock['delta_rmse']:+.9f}. 사전 판정식은 계절 < day. 독립 math.fsum/NumPy 검산 PASS.

| 검증기 | 시드 | day v2 RMSE | 계절 v2 RMSE | 개선 |
|---|---|---|---|---|
{chr(10).join(rows)}

DIAG10 검산 통계: {json.dumps(va['independent_bootstrap'],ensure_ascii=False)}

후반·온실별 진단: {json.dumps(result['diagnostics'],ensure_ascii=False)}

신뢰도: 실행·재현 수치는 높음. 일반화 해석은 중간이다. DC4는 DC3 결과를 본 뒤 설계했고 공개 검증기를 반복 사용했다. 잠금40일도 같은 원자료의 표본이며 완전히 새로운 데이터가 아니다. PFN문맥은 R3시드 사이에 공유했다. 쿼리/미래 입력 변조 불변 검사를 통과했으며, 날씨 변환은 학습 날만으로 맞췄다. 실제 달력을 복원했다는 인과 증명은 아니다.

잠금 확인은 이미 소비됐으므로 이40일을 다시 후보 선택이나 튜닝에 사용하지 않는다. 결과와 무관하게 기준·비중·시드를 사후 변경하지 않았다. 기존 v2 전달본과 다른 AI 파일은 수정하지 않았고 새 후보 제출 CSV/ZIP을 만들지 않았다. 원시 잠금 라벨을 재열람하지 않고 저장된 첫 채점과 예측 해시만으로 이 보고서를 작성했다.

근거: PROTOCOL.md, et_replication_result.json, *_independent_verification.json, locked_confirmation_result.json. 사전등록1484168/235e252, 사용자 직접 추가 요청에 따른 순서9cb84ab.
'''
    with (HERE/'DC4_판정보고서_v1.md').open('x',encoding='utf-8') as stream:stream.write(text)
    record={'status':'VERIFIED','replication_pass':True,'public_gate_pass':public,
            'locked_gate_pass':lock['locked_gate_pass'],'candidate_adopted':adopted,
            'locked_40_consumed':True,'platform_submission':False,
            'evidence_sha256':{p.name:sha(p) for p in [target,HERE/'v2_integration_independent_verification.json',HERE/'DC4_판정보고서_v1.md']}}
    with (HERE/'DC4_판정_v1.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
