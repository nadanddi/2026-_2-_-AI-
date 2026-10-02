"""Read-only audit of already saved seasonal R3 train vs validation error."""
from pathlib import Path
import sys,json,math,hashlib
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
CACHE=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
STAGE=ROOT/'집/코덱스/local/ec_submission10_season_20261002_v2/artifact/stage'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finish(values,ids,low,high):
    groups={};out=np.empty(len(ids))
    for i,(key,value) in enumerate(zip(ids,values)):groups.setdefault(key[:7],[]).append((key,i,float(value)))
    for rows in groups.values():
        running=0.
        for j,(_,i,value) in enumerate(sorted(rows)):
            running+=value;out[i]=np.clip(.5*value+.5*running/(j+1),low,high)
    return out
def rmse(y,p):
    a=math.sqrt(math.fsum((float(u)-float(v))**2 for u,v in zip(y,p))/len(y))
    b=float(np.sqrt(np.mean((y-p)**2)))
    assert abs(a-b)<1e-12;return a
def main():
    oof=pd.read_csv(CACHE/'v2_integration_oof.csv',float_precision='round_trip')
    truth=oof[oof.validator.eq('DIAG10')].groupby('row_id').sub_ec.first().to_dict()
    records=[]
    for fold in range(10):
        for seed in [7,101,2024]:
            file=CACHE/f'DIAG10_{fold}_r3_{seed}.npz'
            meta=json.loads(file.with_suffix('.json').read_text(encoding='utf-8'))
            assert sha(file)==meta['prediction_sha256']
            with np.load(file) as z:
                ids=z['train_row_id'];y=np.array([truth[k] for k in ids]);qy=z['sub_ec']
                tr=rmse(y,finish(z['train_raw_r3'],ids,y.min(),y.max()))
                va=rmse(qy,finish(z['raw_r3'],z['row_id'],y.min(),y.max()))
                assert abs(tr-meta['train_post_rmse'])<1e-12 and abs(va-meta['post_validation_rmse'])<1e-12
                records.append({'fold':fold,'seed':seed,'training_rows':len(y),'validation_rows':len(qy),'train_post_rmse':tr,'validation_post_rmse':va})
    result={'status':'PASS','scope':'Seasonal R3 component only, existing DIAG10 10 folds x 3 seeds; no new fit; no locked labels','method':'saved predictions, independent serial prefix average, math.fsum and NumPy','cells':records,'train_fold_rmse_mean':float(np.mean([r['train_post_rmse'] for r in records])),'validation_fold_rmse_mean':float(np.mean([r['validation_post_rmse'] for r in records])),'training_variance_explanation':'ET leaf1 can closely fit training labels; training error does not measure generalization. PFN training error was not evaluated.'}
    target=HERE/'generalization_gap_v1.json';assert not target.exists();target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    doc=f'''# 과적합 검산 부록 · 2026-10-02 집 코덱스

기존 계절 R3의 DIAG10 10폴드×3시드30칸에서 저장된 학습/검증 예측을 읽고, 독립 순차 당일 평균과 math.fsum/NumPy로 각 RMSE를 재계산했다. 평균 fold 학습 RMSE {result['train_fold_rmse_mean']:.9f}, 평균 fold 검증 RMSE {result['validation_fold_rmse_mean']:.9f}. 표본은 폴드마다 다르고 학습 행이 여러 번 등장하므로 이30칸 평균은 통합 OOF RMSE와 다르다. 모든30칸이 기존 cache metadata와1e-12 이내였다.

ET leaf1은 학습 정답을 매우 잘 맞추므로 학습 오차가 일반화 근거가 될 수 없다. 이 수치는 계절 R3 구성요소에 한정되며 전체 계절 v2 또는 최종400일 fit의 학습 성능을 주장하지 않는다. PFN의 in-sample 점수는 평가하지 않았다. 원시 잠금40일 라벨은 열지 않았고 공개 OOF의 비잠금 정답만 대조했다. 최종400일 fit은 제출 예측용이며 새 홀드아웃 성능을 측정하지 않았다.

재현/행 정합성은 높음, 새 평가일 일반화는 중간 신뢰도다. 공개 검증 반복 사용, DC3 결과 뒤 DC4 설계, 기존5회차 리더보드 모순은 설명자료의 남은 위험으로 유지한다. 세 시드의 PFN문맥 공유를 독립 반복실험으로 해석하지 않는다.
'''
    note=STAGE/'과적합_검산_부록.md';assert not note.exists();note.write_text(doc,encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cells'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
