import run as R
import csv,math,json,collections
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
import cold_v5
np=R.np
HERE=R.HERE
OUT2=R.ROOT/'집/코덱스/local/temp_season_base_20261003_v1'
def main():
    results=[json.loads((p/'result.json').read_text(encoding='utf-8')) for p in [R.OUT,OUT2]]
    labels={}
    with (R.Path(R.env.DATA)/'train_y.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):labels[row['row_id']]=float(row['sub_temp'])
    checks=[];pavdiff=0.
    for path,result in zip([R.OUT,OUT2],results):
        with (path/'oof.csv').open(encoding='utf-8',newline='') as f:
            rows=list(csv.DictReader(f))
        assert all(float(r['sub_temp'])==labels[r['row_id']] for r in rows)
        for fold in result['folds']:
            for farm in ['F13','F47']:
                note=fold['season_fit'][farm];x=np.array(note['anchor_day']);y=np.array(note['anchor_season_raw'])
                fit=IsotonicRegression(increasing=True).fit(x,y).predict(x)
                diff=float(np.max(np.abs(fit-np.array(note['anchor_season_pav']))));pavdiff=max(pavdiff,diff);assert diff<1e-10
        checks.append(dict(path=str(path),label_check_rows=len(rows),labels='PASS'))
    # Replay one BASE residual pair from raw inputs; no Nyström cache dependency.
    R.common.load_raw=R.TM.masked_loader
    try:lab,ct,phc=R.TM.build_world()
    finally:R.common.load_raw=R.TM.ORIG;R.harness._CACHE.clear()
    tm,vm=R.common.split_mask(lab,R.diag_folds(lab)[0]);tr,va=lab[tm].copy(),lab[vm].copy()
    w=R.TF.row_weights(lab,.2,w_noisy=.2)
    raw,_,_=R.TM.ORIG();vec=R.season.vectors(raw[raw.farm.isin(['F13','F47'])]);td=tr[['farm','day']].drop_duplicates();qd=va[['farm','day']].drop_duplicates().reset_index(drop=True)
    sm,sq,_=R.season.mapping(td,qd,{key:vec[key] for key in map(tuple,td.to_numpy())});tr['season']=[sm[(f,int(d))] for f,d in zip(tr.farm,tr.day)]
    qm={(f,int(d)):s for (f,d),s in zip(qd.itertuples(index=False,name=None),sq)};va['season']=[qm[(f,int(d))] for f,d in zip(va.farm,va.day)]
    imp=R.SimpleImputer(strategy='median').fit(tr[phc]);lin=LinearRegression().fit(imp.transform(tr[phc]),tr.sub_temp.to_numpy(),sample_weight=w[tm])
    btr,bva=lin.predict(imp.transform(tr[phc])),lin.predict(imp.transform(va[phc]));resid=tr.sub_temp.to_numpy()-btr
    cache=dict(np.load(OUT2/'DIAG10_0.npz',allow_pickle=True));diffs=[];gap=[]
    for bs,cs in [(7,726),(101,727)]:
        cold_v5.SEED=bs;m0=cold_v5.lgbh();m1=cold_v5.lgbh();cols=[('season' if c=='day' else c) for c in ct]
        m0.fit(tr[ct],resid,sample_weight=w[tm]);m1.fit(tr[cols],resid,sample_weight=w[tm])
        delta=.65*(m1.predict(va[cols])-m0.predict(va[ct]));diff=float(np.max(np.abs(delta-(cache[f'season_{cs}']-cache[f'base_{cs}']))));assert diff<1e-10;diffs.append(diff)
        train=R.rmse(btr+m1.predict(tr[cols]),tr.sub_temp.to_numpy());val=R.rmse(bva+m1.predict(va[cols]),va.sub_temp.to_numpy())
        assert abs(train-float(cache[f'train_rmse_{cs}']))<1e-10 and abs(val-float(cache[f'val_rmse_{cs}']))<1e-10
        gap.append(dict(seed=bs,train_rmse=train,validation_rmse=val,scope='DIAG10 fold0 BASE physics+residual only'))
    audit=dict(status='PASS',label_checks=checks,pav_sklearn_maxdiff=pavdiff,retrain_delta_maxdiff=max(diffs),train_validation_gap=gap)
    (HERE/'final_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    verifs=[json.loads((HERE/f).read_text(encoding='utf-8')) for f in ['verification.json','verification_base.json']]
    text=['# W30G 계절 정보 적용 결과 · 2026-10-03 집 코덱스','','사용자가 온도 개선을 직접 지시하여 이번에 코덱스가 온도 실험을 수행했다. EC 계절 지표를 온도 잔차 트리에 적용하되 저온 물리식과 혼합비율은 보존했다.','','| 적용 위치 | DIAG10 RMSE 변화 | EXT10 | EXT12 | DIAG p_worse | 판정 |','|---|---|---|---|---|---|']
    for name,v in zip(['T-S1 Codex 잔차 트리','T-S2 BASE 잔차 트리'],verifs):
        ranges=[]
        for validator in ['DIAG10','EXT10','EXT12']:
            values=[x['delta_pct'] for x in v['cells'] if x['validator']==validator];ranges.append(f'{min(values):+.3f}~{max(values):+.3f}%')
        ps=[x['p_worse'] for x in v['cells'] if x['validator']=='DIAG10'];text.append(f'| {name} | '+ ' | '.join(ranges)+f' | {min(ps):.5f}~{max(ps):.5f} | {v["verdict"]} |')
    text+=['','## 검증 범위','- DIAG10 400일9600행, EXT10/EXT12 전체 저온일 홀드아웃, +/-1일 buffer. 2시드 조합×2 PFN문맥계열×3검증기=후보별12칸. BASE/PFN 저장 예측은 그대로 사용하며 잔차 변경 위치만 재학습.','- 원본 W30G BASE/CODEX 재학습 최대차0. 원시 train_y 대조, csv/math.fsum RMSE·멤버혼합·20k 부트스트랩 독립검산 PASS. 등위회귀 sklearn/PAV 비교 및 첫 fold 재학습 재현은 final_audit.json.','- 두 후보 시험에 대한 본페로니 기준 p<.0125/97.5% CI 및 전칸 개선. T-S1은 최초 k1 p<.025에도 실패. T-S2는 T-S1 결과를 보고 적용 위치를 바꾼 순차 실험이며 별도 독립 홀드아웃은 사용하지 않았다.','- season은 각fold 학습날씨만 fit, 검증일 farm/day 보간. 검증 날씨·평가 날씨·EC정답·리더보드로 특징/비중 선택 없음. 원본 모델 입력은 MASK. 새 season 변환은 검증 query 순서 변조 불변(T-S1)·query날씨 키 부재.','- 기존 온도 input-only 가중치의 전체 학습일 rank는 고정 기준선과 동일하며 fold별 재산정하지 않음. 최종3시드평균 제출 모델 자체의 새 점수는 측정하지 않았음.','','## 해석 및 반론','- 두 위치에서 이번 적용이 채택 기준을 넘었는지는 표의 판정으로만 판단한다. 작은 평균 개선만으로 새 모델을 채택하지 않는다.','- 계절 정보가 온도에서 전혀 쓸모없음을 증명한 결과인가? 아니다. 두 잔차 트리의 day 교체만 시험했고 PFN 문맥 변경/계절×물리 상호작용은 시험하지 않았다.','- 일부 저온/후반 구간이 좋아도 전체 규칙을 넘지 못하면 채택하지 않는다. 세그먼트 원시 수치는 verification*.json에 포함했다.','- 재현·산술 신뢰도 높음. 숨은 평가일 일반화는 미확인. 학습/검증 격차는 첫 fold의 BASE 물리+잔차 구성원에 한정되며 전체 W30G의 학습 오차가 아니다.','','## 파일 및 다음','- 사전등록 main f2c21d7(T-S1), 4a86559(T-S2). PROTOCOL*.md/run*.py/verify_v3.py/verify_base_v2.py. 검산 구버전 실패는 npz 문자열 object array 로딩 설정·블록 문자열 정렬 순서 문제였고 학습/후보 기준은 변경하지 않았다.','- 결과는 기존 파일을 수정하지 않고 별도 local/temp_season_20261003_v1, local/temp_season_base_20261003_v1에 저장했다.','- 제출 파일/플랫폼 업로드/EC 잠금 채점 없음. 두 안이 기각이면 W30G 유지하고 이번 적용 실험 종료.']
    (HERE/'결과보고서_v1.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
