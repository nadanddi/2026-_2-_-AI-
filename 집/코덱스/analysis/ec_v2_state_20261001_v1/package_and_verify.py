from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import argparse,json,zipfile,shutil,hashlib,subprocess,os,csv,math
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local/ec_v2_state_20261001_v1';OUT=LOCAL/'artifact';STAGE=OUT/'stage';PRED=OUT/'predictions'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safezip(path,source):
    assert not path.exists()
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(source.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(source).as_posix())
def main():
    a=argparse.ArgumentParser();a.add_argument('--mode',choices=['prepare','verify'],required=True);args=a.parse_args()
    if args.mode=='prepare':
        result=json.loads((LOCAL/'cv_verified.json').read_text(encoding='utf-8'));scores=pd.read_csv(LOCAL/'verified_scores.csv');seeds=pd.read_csv(LOCAL/'verified_seed_scores.csv')
        status='수치 선별 통과, 독립 확인 미실시로 채택 보류' if result['numerical_screen_pass'] else '개선 검증 실패 — 사용자 요청으로 생성한 실험용 모델'
        table=['| 검증 | 기존v2 RMSE | 새EC v2 RMSE |','|---|---:|---:|']
        for name in ['DIAG10','A','B','EXT10','EXT12']:
            g=scores[scores.validator.eq(name)].set_index('model');table.append(f'|{name}|{g.loc["baseline","rmse"]:.6f}|{g.loc["candidate","rmse"]:.6f}|')
        explanation=f'''# EC 전용 v2_state_v1 — 팀원 전달 자료

2026-10-01 집·코덱스. 상태: **{status}**. 공식 평가 점수는 미측정, 플랫폼 제출하지 않았다. 온도 예측은 포함하지 않는다.

## 파일과 결합 방법

`ec_v2_state_v1.csv`는 평가1440행, 열은 `row_id,sub_ec` 두 개다. 원래 sample_submission ID 순서다. 온도팀은 row_id로 자신의 sub_temp와 결합한다. 행번호만으로 붙이지 않는다. 이 CSV 자체는 대회 요구3열의 완성 제출물이 아니다.

재현 ZIP에 `merge_ec.py`를 넣었다. 팀원이 `python merge_ec.py --temperature 온도.csv --ec ec_v2_state_v1.csv --output 결합_새이름.csv`로 실행하면 ID 유일성/동일 집합/1440행/유한값을 검사하고 온도 CSV 순서의 row_id,sub_temp,sub_ec를 만든다. 기존 온도 CSV에 EC 열이 있으면 해당 열을 사용하지 않고 새 EC를 row_id로 붙인다.

## 모델

같은 온실의 예측일보다 앞선 가장 최근 **사용 가능한 공개 학습일** 전체 EC 평균과 그 날23시부터 경과시간을 기존38열 ExtraTrees에 더했다. 과거 기록이 같은 실제 출처라는 보장은 없다. 자기일·미래일·다른온실·검증일·잠금40일±1 정답은 특징에서 제외했다. NaN은 학습폴드 median 대체.

기존v2=.8R3+.2TabPFN, R3=.6ET+.3Tweedie LGB+.1MLP. 새모델은 seed별 `clip(v2_final+.48*(ET40_final−ET38_final),학습EC범위)` 후 seed7/101/2024평균이다. .48=.8*.6 고정, 튜닝없음. final은 .5현재+.5당일현재까지예측평균 후clip. raw ET교체와clip경계에서다를 수 있는 **후처리 변화량 모델**이다. 기본38열/원래혼합비/모델수 유지.

ET600trees/leaf1, LGB800trees/learning_rate.03/Tweedie1.5, MLP128-64/earlystop, TabPFN v2로컬checkpoint/CPU float32/2000학습문맥4개×4estimators. 모델의 전체학습점수는미측정. 검증과학습점수차이를새모델의과적합없음으로주장하지않는다.

## 검증

{chr(10).join(table)}

A/B는중첩폴드발생단위 pooled RMSE이며 고유행평균점수는 `verified_scores.csv`에별도있다. DIAG10비잠금360일8640행이정확히1회, A/B각5/EXT각1 등22fold3seed. 시드×검증기15칸개선: {int(seeds.delta.lt(0).sum())}/15. DIAG10후보−기존 차이95%구간 [{result['difference_ci95'][0]:+.6f},{result['difference_ci95'][1]:+.6f}], block bootstrap20000 p_worse={result['p_worse']:.5f}. 가설1개, 전시드×전검증기개선/CI<0/p<.025가수치선별기준. 독립미사용확인미실시·기존자료반복분석이므로정식채택없음. 새특징을도입했다고성능개선이나.05달성을보장하지않는다.

최종모델도잠금40일±1제외학습. ZIP의 train_y는EC전용비잠금8640행만이며온도정답/잠금정답없음. 공개train_y 평가특징활용은사용자허용범위. 특징은과거만참조. 최종잠금채점0.

## 재현

Python3.12, `requirements.txt`의 라이브러리 버전 사용. 로컬checkpoint가 ZIP에있어모델다운로드필요없음. ZIP을새폴더에풀고 `python reproduce.py --output 새출력폴더` 실행. 출력폴더가이미있으면덮어쓰지않는다. CPU추론으로시간이걸린다. CUDA불필요.

이PC의별도설치된third-party라이브러리를쓰려면PowerShell `$env:EC_DEPENDENCY_PATH='C:\\work\\farmai\\.analysis-tools\\extra'`를지정한다. 라이브러리가일반Python환경에설치되어있으면이설정불필요. 실행소스·데이터·checkpoint는ZIP내파일만사용한다. WindowsMSVC DLL동봉. 실제버전/학습일수/출력해시는 `manifest.json` 참조.

깨끗한ZIP추출후전체모델재학습·예측비교결과는동봉 `reproduction_verification.json`에기록한다. 원시EC라벨정합성/22캐시/독립RMSE산술/날짜·온실·미래불변검사PASS. 재현검증의허용최대예측차이는1e−8이다.

## 한계

최근달력상학습일은동일출처전날로확인된연결이아니다. 기존검증은엄격한전진시계열전체평가가아니며EXT는학습입력패널만의기준이다. A/B반복일을독립표본으로간주하지않는다. 관수기작·센서정의·공식점수예상은확정하지않는다. 사용자요청생성과모델채택을구분한다.
'''
        assert not (OUT/'설명자료_ec_v2_state_v1.md').exists()
        (OUT/'설명자료_ec_v2_state_v1.md').write_text(explanation,encoding='utf-8');shutil.copyfile(OUT/'설명자료_ec_v2_state_v1.md',STAGE/'설명자료_ec_v2_state_v1.md')
        for n in ['cv_verified.json','verified_scores.csv','verified_seed_scores.csv','verified_segments.csv']:shutil.copyfile(LOCAL/n,STAGE/n)
        for n in ['ec_v2_state_v1.csv','manifest.json','past_state_audit.csv']:shutil.copyfile(PRED/n,STAGE/n)
        shutil.copyfile(PRED/'ec_v2_state_v1.csv',OUT/'ec_v2_state_v1.csv')
        temp=OUT/'reproduction_check_input_v1.zip';safezip(temp,STAGE)
        extract=OUT/'reproduction_check';assert not extract.exists()
        with zipfile.ZipFile(temp) as z:
            assert z.testzip() is None
            for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
            z.extractall(extract)
        print(f'PACKAGE CHECK INPUT READY {extract}',flush=True)
    else:
        actual=pd.read_csv(PRED/'ec_v2_state_v1.csv');replay=pd.read_csv(OUT/'reproduction_check/replay/ec_v2_state_v1.csv');sample=pd.read_csv(STAGE/'data/sample_submission.csv')
        assert list(actual.columns)==list(replay.columns)==['row_id','sub_ec'] and len(actual)==1440
        assert actual.row_id.is_unique and actual.row_id.tolist()==replay.row_id.tolist()==sample.row_id.tolist()
        assert np.isfinite(actual.sub_ec).all() and actual.sub_ec.ge(0).all()
        diff=float(np.max(np.abs(actual.sub_ec-replay.sub_ec)));assert diff<=1e-8
        audit=pd.read_csv(PRED/'past_state_audit.csv');assert audit.state_source_day.notna().all() and (audit.state_source_day<audit.day).all() and audit.prev_public_gap_hours.gt(0).all()
        # Independent csv arithmetic checks three output quantities.
        with (PRED/'ec_v2_state_v1.csv').open(newline='',encoding='utf-8-sig') as f:values=[float(r['sub_ec']) for r in csv.DictReader(f)]
        assert len(values)==len(actual) and math.isclose(math.fsum(values)/len(values),float(actual.sub_ec.mean()),abs_tol=1e-12) and math.isclose(min(values),float(actual.sub_ec.min()),abs_tol=1e-12)
        with (OUT/'reproduction_check/replay/ec_v2_state_v1.csv').open(newline='',encoding='utf-8-sig') as f:v2=[float(r['sub_ec']) for r in csv.DictReader(f)]
        independent=max(abs(a-b) for a,b in zip(values,v2));assert math.isclose(diff,independent,abs_tol=1e-12)
        # Every file extracted from the candidate ZIP must still match source stage.
        for p in STAGE.rglob('*'):
            if p.is_file():assert sha(p)==sha(OUT/'reproduction_check'/p.relative_to(STAGE))
        r={'status':'PASS','rows':1440,'columns':['row_id','sub_ec'],'sample_order':'PASS','finite_nonnegative':'PASS','prior_state_causality':'PASS','clean_extraction_full_retraining':'PASS','max_absolute_prediction_difference':diff,'independent_csv_arithmetic':'PASS','source_files_unchanged':'PASS','original_csv_sha':sha(PRED/'ec_v2_state_v1.csv'),'replayed_csv_sha':sha(OUT/'reproduction_check/replay/ec_v2_state_v1.csv'),'candidate_adopted':False,'platform_submitted':False}
        assert not (OUT/'reproduction_verification.json').exists();(OUT/'reproduction_verification.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copyfile(OUT/'reproduction_verification.json',STAGE/'reproduction_verification.json')
        final=OUT/'EC_v2_state_v1_재현패키지.zip';safezip(final,STAGE)
        with zipfile.ZipFile(final) as z:assert z.testzip() is None
        (OUT/'delivery_manifest.json').write_text(json.dumps({'status':'PASS','files_sha256':{n:sha(OUT/n) for n in ['ec_v2_state_v1.csv','EC_v2_state_v1_재현패키지.zip','설명자료_ec_v2_state_v1.md','reproduction_verification.json']},'ec_only':True,'platform_submitted':False},ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(r,ensure_ascii=False,indent=2));print(final)
if __name__=='__main__':main()
