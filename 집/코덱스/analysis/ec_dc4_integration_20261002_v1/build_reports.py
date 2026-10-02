"""독립 검산 및 잠금 단회 결과가 모두 존재할 때만 최종 연구 보고서를 생성."""
from pathlib import Path
import json,sys,hashlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
K2=HERE.parent/'ec_stage2_tabpfn_20261002_v2'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def table(rows,columns):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(str(r[c]) for c in columns)+' |' for r in rows])

def main():
    ev=read(HERE/'et_replication_independent_verification.json')
    vv=read(HERE/'v2_integration_independent_verification.json')
    lock=read(HERE/'locked_confirmation_result.json')
    kv=read(K2/'independent_verification_v2.json')
    assert all(r['status']=='PASS' for r in [ev,vv,kv]) and lock['status']=='CONSUMED'
    et=read(HERE/'et_replication_result.json');v2=read(HERE/'v2_integration_result.json');k=read(K2/'result.json')
    public=vv['direction_pass'] and vv['confidence_pass'];adopted=public and lock['locked_gate_pass']
    decision='全조건 통과 연구 후보' if adopted else '기각: 정식 후보 조건 미충족'
    decision=decision.replace('全','모든 ')
    comparisons=[]
    for z in v2['ensemble']:
        if z['arm']=='season_v2':
            b=next(r for r in v2['ensemble'] if r['validator']==z['validator'] and r['arm']=='v2')
            comparisons.append({'검증기':z['validator'],'기존 v2':f'{b["rmse"]:.9f}','계절 v2':f'{z["rmse"]:.9f}','변화%':f'{100*(z["rmse"]/b["rmse"]-1):+.3f}','발생행':z['n']})
    individual=[{'검증기':r['validator'],'시드':r['seed'],'기준RMSE':f'{r["reference_rmse"]:.9f}','계절RMSE':f'{r["candidate_rmse"]:.9f}','개선':r['improved']} for r in vv['independent_scores']]
    report=f'''# EC DC4 독립 재현 및 전체 v2 통합 결과 · 2026-10-02 집 코덱스

판정: **{decision}**. 독립 ET 재현 PASS, 공개 통합 {'PASS' if public else 'FAIL'}, 잠금 단회 {'PASS' if lock['locked_gate_pass'] else 'FAIL'}. 모든 조건을 통과해야 채택하는 사용자 결정 그대로 판정했다. 플랫폼 제출 및 새 후보 제출물 생성은 하지 않았다.

독립 계절 변환은 NumPy 및 직접 PAV로 구현했으며 원본 sklearn 등위 회귀와 전 폴드에서 최대차 1e-10 이내였다. ET 예측은 전22폴드×3시드에서 원본 OOF와 최대차 {max(r['ET_max_gap'] for r in et['independent_reproduction']):.3g}, 사전 허용1e-8 이내다. 쿼리/미래 외기 변조와 쿼리 행순서 불변 검사 PASS. 원본 cal/deep_cal_9를 모델·변환에 쓰지 않았다. 원본 ET는 {sum(r['improved'] for r in ev['independent_scores'])}/15칸 개선으로, 재현 성공과 채택 성공을 구분한다.

## 공개 v2 통합

R3 전체 ET/LGB/MLP 및 PFN 입력에서 day 대신 season을 끝 열에 넣었다. 기록 day는 분할·후처리에 보존했다. 원시 .8R3+.2PFN4, R3시드7/101/2024 및 PFN문맥1..4, 기존 동일 fit/문맥 행·후처리. 22폴드27720발생행×3시드83160행이다. DIAG10은 비잠금360일8640고유행이다. A/B는 반복 발생행 pooled RMSE이며 아래 시드평균표는 진단용이다.

{table(comparisons,['검증기','기존 v2','계절 v2','변화%','발생행'])}

{table(individual,['검증기','시드','기준RMSE','계절RMSE','개선'])}

공개 판정은 {sum(r['improved'] for r in vv['independent_scores'])}/15칸 개선 및 세 DIAG10 검정 모두 p<.0125/97.5% CI상한<0. 온실층화80개5일블록20000회, seed918. 이번 E40/DC4 두 후보에 보수적 보정을 고정했다.

{table(vv['independent_bootstrap'],['seed','p_worse','ci_low','ci_high'])}

후반·온실별 구성원 진단: {json.dumps(v2['diagnostics'],ensure_ascii=False)}

## 잠금 40일 단회 확인

기존 final fit과 같은306일7344행, 잠금±1purge. R3세시드평균/PFN네문맥평균을 고정해 두 모델의 960행 예측을 모두 저장한 후 정답을 처음 숫자로 읽었다. 소비 기록 LOCK_CONSUMED_ONCE.json으로 재채점 금지. 사전식은 계절 v2 RMSE < 같은 day v2 RMSE다. day={lock['day_rmse']:.9f}, 계절={lock['season_rmse']:.9f}, 차={lock['delta_rmse']:+.9f}. math.fsum/NumPy 독립 산술 검산 PASS. 공개/잠금 결과와 무관하게 같은40일을 새 모델 선택에 다시 쓰지 않는다.

## 검산과 한계

CSV 표준 라이브러리 재파싱·math.fsum RMSE, 부트스트랩 전20000회, 고유행/전체폴드/시드 및 캐시 SHA/문맥 인덱스를 확인했다. 독립 공개 RMSE 최대차 {vv['max_RMSE_crosscheck_gap']:.3g}. 신뢰도: 실행·재현 수치는 높음. 일반화 해석은 중간이다. DC4는 이전 DC3 결과를 보고 설계됐고 공개 검증기를 반복 사용했다. 잠금40일도 같은 원자료이며 별도 수집 데이터가 아니다. PFN문맥은 세 R3시드에서 공유되므로 완전히 독립인 세 실험으로 해석하지 않는다. 1차 쿼리 경계 보간·2차 최근접 대체의 원본 동작을 그대로 재현했다. season이 알려진 운영 계절이라는 인과 증명은 없다.

근거: PROTOCOL.md, et_replication_result.json, v2_integration_result.json, *_independent_verification.json, locked_confirmation_result.json. 코드 사전등록1484168/235e252. 사용 입력은 학습 MASK14열, 같은 온실 현재·과거 인과 특징, 학습 날만의 외기 변환. 리더보드 비중 선택/test_X 통계/외부조회 없음. 기존 K1 전달본을 수정하지 않았다.
'''
    with (HERE/'결과보고서_v1.md').open('x',encoding='utf-8') as stream:stream.write(report)
    kr=[{'검증기':r['validator'],'E40_8':f'{r["E40_8"]:.9f}','새 v2':f'{r["v2_fresh"]:.9f}','기존 v2':f'{r["v2_fixed"]:.9f}'} for r in k['ensemble_diagnostic_only']]
    kreport=f'''# EC K2 E40_8 새 시드 확인 및 K3 자료 생성 · 2026-10-02 집 코덱스

K2 판정: {'공개 검증 통과 후보' if kv['public_gate_pass'] else '기각'}. 정확한 E40_8=.6R3+.4PFN8을 FULL38(실내 평균3열 추가 아님), 새R3시드401/402/403 및 PFN문맥5..12로 재확인했다. 고정 기준은 v2_fresh=.8R3+.2PFN5..8, 기존3시드평균 v2_fixed다. 두 기준×3시드×5검증기 총30칸 중 {sum(r['improved'] for r in kv['scores'])}칸 개선. 모든30칸 및 DIAG6검정 p<.0125/97.5%CI상한<0 기준을 유지했다. 후반·EXT 실패를 시드평균으로 덮지 않았다.

{table(kr,['검증기','E40_8','새 v2','기존 v2'])}

{table(kv['bootstrap'],['seed','reference','p_worse','ci_low','ci_high','passed'])}

K3는 별도 후보가 아닌 공개 OOF 자료다. 22폴드27720발생행, PFN문맥5~8 원시열과5~12 8문맥·평균4/8·R3시드별 원시 및 후처리 예측은 집/코덱스/local/ec_stage2_tabpfn_20261002_v2/oof_wide.csv 및 .npz, oof_long.csv에 있다. 176PFN 및66R3캐시 전수해시·문맥행검사PASS. 비잠금8640원시라벨 일치, 독립 CSV/수동fsum RMSE 최대차 {kv['max_independent_rmse_gap']:.3g}, 부트스트랩 전수 재계산PASS.

신뢰도: 재현·검산 수치는 높음, 숨은 평가 개선의 해석은 중간. 공개 검증 반복 사용과 PFN문맥 공유의 한계가 있다. 효과 크기컷 없이 원사용자 기준대로 판정했다. 잠금40일은 이 K2실험에서 채점하지 않았다. 같은 원자료로 DC4 단회 확인이 별도로 완료됐으므로 이후 해당 잠금은 재사용하지 않는다. 실제 제출/온도모델/새 후보 전달본 생성 없음.

근거: PROTOCOL.md, result.json, independent_verification_v2.json, completion.json. 코드/기준50f86fa, 독립 검산e9557b0.
'''
    with (K2/'결과보고서_v1.md').open('x',encoding='utf-8') as stream:stream.write(kreport)
    final={'DC4_ET_replication':'PASS','DC4_public_gate_pass':public,'DC4_locked_gate_pass':lock['locked_gate_pass'],
           'DC4_candidate_adopted':adopted,'K2_public_gate_pass':kv['public_gate_pass'],'K3_complete':True,
           'locked_40_consumed':True,'submission_created':False,'platform_submission':False,
           'evidence_sha256':{str(p):sha(p) for p in [HERE/'v2_integration_independent_verification.json',HERE/'locked_confirmation_result.json',K2/'independent_verification_v2.json']}}
    with (HERE/'final_review_v1.json').open('x',encoding='utf-8') as stream:json.dump(final,stream,ensure_ascii=False,indent=2)
    print(json.dumps(final,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
