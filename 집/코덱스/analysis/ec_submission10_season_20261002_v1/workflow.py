"""Create new artifacts, actually train twice, verify, then package. No overwrites."""
from pathlib import Path
import sys,os,csv,json,shutil,hashlib,zipfile,subprocess,math,importlib.util,contextlib,io
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import env_extra
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
OUT=ROOT/'집/코덱스/local/ec_submission10_season_20261002_v1/artifact'
STAGE=OUT/'stage'
OLD=ROOT/'집/코덱스/local/ec_v2_base_delivery_20261002_v1/artifact/stage'
PROOF=ROOT/'집/코덱스/analysis/ec_dc4_integration_20261002_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def archive(path,folder):
    assert not path.exists()
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(folder).as_posix())
def subset(source,target,cols,farms=None):
    with source.open(encoding='utf-8-sig',newline='') as f,target.open('w',encoding='utf-8-sig',newline='') as out:
        r=csv.DictReader(f);w=csv.DictWriter(out,fieldnames=cols,lineterminator='\n');w.writeheader()
        for row in r:
            if farms and row['row_id'][:3] not in farms:continue
            w.writerow({k:row[k] for k in cols})
def prepare():
    assert not OUT.exists(),'New version path required'
    (STAGE/'data').mkdir(parents=True)
    for n in ['model.py','season.py','reproduce.py','PROTOCOL.md']:shutil.copyfile(HERE/n,STAGE/n)
    for n in ['env.py','env_extra.py','requirements.txt','tabpfn-v2-regressor.ckpt']:shutil.copyfile(OLD/n,STAGE/n)
    shutil.copytree(OLD/'dlls',STAGE/'dlls')
    raw=['in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
    weather=['out_temp','out_hum','out_rad','out_wspd'];data=Path(env.DATA)
    subset(data/'train_X.csv',STAGE/'data/train_X.csv',['row_id']+weather+raw,['F13','F47'])
    subset(data/'train_y.csv',STAGE/'data/train_y.csv',['row_id','sub_ec'],['F13','F47'])
    subset(data/'test_X.csv',STAGE/'data/test_X.csv',['row_id']+raw)
    shutil.copyfile(data/'sample_submission.csv',STAGE/'data/sample_submission.csv')
    x=read(STAGE/'data/train_X.csv');y=read(STAGE/'data/train_y.csv');tx=read(STAGE/'data/test_X.csv')
    assert len(x)==len(y)==9600 and len(tx)==1440
    assert len({r['row_id'][:7] for r in y})==400
    assert {r['row_id'] for r in x}=={r['row_id'] for r in y}
    assert not ({r['row_id'] for r in x}&{r['row_id'] for r in tx})
    mask=read(data/'test_X.csv')
    assert all(any(row[c].strip() for row in mask) for c in raw)
    excluded=[c for c in mask[0] if c!='row_id' and c not in raw]
    assert all(not row[c].strip() for row in mask for c in excluded)
    ledger=json.loads((PROOF/'DC4_판정_v1.json').read_text(encoding='utf-8'))
    assert ledger['status']=='VERIFIED' and ledger['candidate_adopted']
    for n,h in ledger['evidence_sha256'].items():assert sha(PROOF/n)==h
    manifest={'status':'PREPARED','original_inputs_sha256':{n:sha(data/n) for n in ['train_X.csv','train_y.csv','test_X.csv','sample_submission.csv']},'training_rows':9600,'training_days':400,'query_rows':1440,'query_days':60,'temperature':'ALL_BLANK_BY_USER_REQUEST','test_mask_excluded_columns':excluded,'candidate_evidence_sha256':ledger['evidence_sha256'],'stage_files_sha256':{p.relative_to(STAGE).as_posix():sha(p) for p in STAGE.rglob('*') if p.is_file()},'platform_submission':False}
    save(OUT/'preparation_manifest.json',manifest)
    print('Prepared full-training EC-only package, sub_temp blank.',flush=True)
def run(folder,output):
    variables=os.environ.copy();variables['PYTHONPATH']='';variables['PYTHONIOENCODING']='utf-8'
    variables['EC_CORE_DEPENDENCY_PATH']=str(ROOT/'.analysis-tools/python')
    variables['EC_DEPENDENCY_PATH']=str(ROOT/'.analysis-tools/extra')
    command=[sys.executable,'-B','-u',str(folder/'reproduce.py'),'--output',str(output)]
    subprocess.run(command,env=variables,cwd=folder,check=True)
def independent_season():
    source=ROOT/'집/클로드/research/ec2_DC4_exact_twin_anchor_v1.py'
    spec=importlib.util.spec_from_file_location('submission10_claude_readonly',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    x=pd.read_csv(STAGE/'data/train_X.csv');tx=pd.read_csv(STAGE/'data/test_X.csv')
    def days(frame):
        return pd.DataFrame({'farm':frame.row_id.str[:3],'day':frame.row_id.str[4:7].astype(int)}).drop_duplicates().sort_values(['farm','day']).reset_index(drop=True)
    td,qd=days(x),days(tx)
    with contextlib.redirect_stdout(io.StringIO()):ts,qs=module.season_index(td,qd,module.weather_vectors(x))
    a=pd.read_csv(OUT/'source/training_day_to_season.csv',float_precision='round_trip')
    b=pd.read_csv(OUT/'source/evaluation_day_to_season.csv',float_precision='round_trip')
    train_gap=max(abs(row.season-ts[(row.farm,row.day)]) for row in a.itertuples())
    eval_gap=float(np.max(np.abs(b.season.to_numpy()-qs)))
    assert train_gap<=1e-9 and eval_gap<=1e-9
    return {'status':'PASS','source_sha256':sha(source),'method':'Claude pivot weather vectors + sklearn IsotonicRegression vs independent PAV','training_days':len(a),'evaluation_days':len(b),'training_max_difference':train_gap,'evaluation_max_difference':eval_gap}
def verify_public():
    source=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv'
    proof=json.loads((PROOF/'v2_integration_independent_verification.json').read_text(encoding='utf-8'))
    assert sha(source)==proof['csv_sha256']
    d=pd.read_csv(source,float_precision='round_trip');cells=[];ensemble=[];segments=[]
    def rmse(y,p):
        a=np.sqrt(np.mean((y-p)**2));b=math.sqrt(math.fsum((float(u)-float(v))**2 for u,v in zip(y,p))/len(y))
        assert abs(a-b)<1e-12;return float(b)
    for (validator,seed),g in d.groupby(['validator','seed']):
        a=rmse(g.sub_ec.to_numpy(),g.v2.to_numpy());b=rmse(g.sub_ec.to_numpy(),g.season_v2.to_numpy())
        assert b<a;cells.append({'validator':validator,'seed':int(seed),'n':len(g),'day_rmse':a,'season_rmse':b})
    assert len(cells)==15
    for validator,g in d.groupby('validator'):
        averaged=g.groupby(['validation_fold','row_id'])[['sub_ec','v2','season_v2']].mean()
        ensemble.append({'validator':validator,'n':len(averaged),'day_rmse':rmse(averaged.sub_ec.to_numpy(),averaged.v2.to_numpy()),'season_rmse':rmse(averaged.sub_ec.to_numpy(),averaged.season_v2.to_numpy())})
    diag=d[d.validator.eq('DIAG10')].groupby('row_id')[['day','sub_ec','v2','season_v2']].mean()
    for name,m in [('전반(day<179)',diag.day.lt(179)),('후반(day>=179)',diag.day.ge(179))]:
        g=diag[m];segments.append({'segment':name,'n':len(g),'days':len(g)//24,'day_rmse':rmse(g.sub_ec.to_numpy(),g.v2.to_numpy()),'season_rmse':rmse(g.sub_ec.to_numpy(),g.season_v2.to_numpy())})
    lock=json.loads((PROOF/'locked_confirmation_result.json').read_text(encoding='utf-8'))
    assert lock['locked_gate_pass'] and lock['labels_read_once']
    lockdir=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/locked_confirmation'
    for n,h in lock['prediction_files'].items():assert sha(lockdir/n)==h
    # Do not re-open locked labels for evaluation. Final-fit labels are separate training inputs.
    return {'status':'PASS','source_sha256':sha(source),'method':'NumPy squared loss and independent math.fsum','cells':cells,'ensemble':ensemble,'segments':segments,'bootstrap_verified_record':proof.get('independent_bootstrap',proof.get('bootstrap')),'lock_once_record':{'rows':960,'day_rmse':lock['day_rmse'],'season_rmse':lock['season_rmse'],'delta_rmse':lock['delta_rmse'],'record_sha256':sha(PROOF/'locked_confirmation_result.json'),'prediction_hashes_verified':True,'rescored':False}}
def finalize():
    orig=OUT/'source';replay=OUT/'clean_check_extract/replay';sample=read(STAGE/'data/sample_submission.csv')
    a=read(orig/'submission_10.csv');b=read(replay/'submission_10.csv')
    assert len(a)==len(b)==len(sample)==1440
    assert list(a[0])==['row_id','sub_temp','sub_ec']
    assert [r['row_id'] for r in a]==[r['row_id'] for r in b]==[r['row_id'] for r in sample]
    assert len({r['row_id'] for r in a})==1440 and all(r['sub_temp']=='' for r in a+b)
    av=np.array([float(r['sub_ec']) for r in a]);bv=np.array([float(r['sub_ec']) for r in b])
    assert np.isfinite(av).all() and np.isfinite(bv).all()
    assert np.array_equal(av.view(np.uint64),bv.view(np.uint64))
    assert sha(orig/'submission_10.csv')==sha(replay/'submission_10.csv')
    for n in ['training_day_to_season.csv','evaluation_day_to_season.csv','season_transform.json']:
        assert sha(orig/n)==sha(replay/n)
    prepared=json.loads((OUT/'preparation_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(STAGE/p)==h and sha(OUT/'clean_check_extract'/p)==h for p,h in prepared['stage_files_sha256'].items())
    mean=math.fsum(av)/len(av);assert math.isclose(mean,float(av.mean()),abs_tol=1e-15)
    season=independent_season();public=verify_public()
    save(OUT/'fresh_public_verification.json',public)
    metadata=json.loads((orig/'manifest.json').read_text(encoding='utf-8'))
    replaymeta=json.loads((replay/'manifest.json').read_text(encoding='utf-8'))
    assert metadata==replaymeta
    verification={'status':'PASS','rows':1440,'sample_id_order_equal':True,'temperature_all_blank':True,'finite_ec':True,'csv_bytes_equal':True,'prediction_bits_equal':True,'max_abs_ec_difference':float(np.max(np.abs(av-bv))),'csv_sha256':sha(orig/'submission_10.csv'),'actual_full_retraining_from_clean_extract':True,'training_rows':9600,'training_days':400,'lock_in_final_fit':True,'lock_rescored':False,'season_tables_bytes_equal':True,'independent_season':season,'mean_ec_numpy':float(av.mean()),'mean_ec_math_fsum':mean,'minimum_ec':float(av.min()),'maximum_ec':float(av.max()),'model_checks':metadata['checks'],'original_inputs':prepared['original_inputs_sha256'],'public_validation_source':public['source_sha256'],'platform_submission':False,'risk_review':[{'challenge':'최종 전체학습 모델도 같은 검증 성능을 보장하는가','answer':'아니다. 기존 누수방지 OOF와 단회 잠금은 레시피 근거이며 전체400일 최종fit은 새 평가정답으로 채점하지 않았다.','confidence':'일반화 중간'},{'challenge':'공개 검증 반복 사용과 5회차 모순','answer':'남은 위험이며 실제 합본 제출 후 확인해야 한다. 리더보드 점수로 비중을 선택하지 않았다.','confidence':'미확정'},{'challenge':'온도 빈칸 CSV를 제출해도 되는가','answer':'팀원 결합용 EC 전달본이다. 팀원이 sub_temp를 채운 뒤 합본과 재현패키지를 다시 대조해야 한다.','confidence':'높음'}]}
    save(OUT/'verification.json',verification)
    rows='\n'.join(f"| {r['validator']} | {r['n']} | {r['day_rmse']:.9f} | {r['season_rmse']:.9f} |" for r in public['ensemble'])
    segment='\n'.join(f"| {r['segment']} | {r['days']} / {r['n']} | {r['day_rmse']:.9f} | {r['season_rmse']:.9f} |" for r in public['segments'])
    lock=public['lock_once_record']
    boot=pd.read_csv(PROOF/'v2_integration_bootstrap.csv')
    assert len(boot)==3 and (boot.p_worse<.0125).all() and (boot.ci_mse_high<0).all()
    bootstrap_lines='\n'.join(f"- 시드 {int(r.seed)}: p_worse={r.p_worse:.5f}, ΔMSE 97.5% CI [{r.ci_mse_low:.9f}, {r.ci_mse_high:.9f}]." for r in boot.itertuples())
    doc=f'''# submission_10 · 8회차 EC 계절 v2 전달본

2026-10-02 집 코덱스. 사용자 요청으로 `sub_temp`는 전부 빈칸이다. 팀원 온도 모델이 채울 EC 전달본이며 플랫폼 업로드는 하지 않았다. 온도 W30G 복사 지시는 최신 빈칸 지시로 대체되었다.

## 파일과 최종 학습

`submission_10.csv`: 1440행, `row_id,sub_temp,sub_ec`, 원본 sample_submission 순서. EC는 결측·무한대가 없으며 온도 빈칸은 의도한 상태다. F13·F47의 정답 있는 전체400일9600행을 최종학습에 사용했다. 이미 한 번 확인한 잠금40일도 학습에 포함했다. 그40일을 다시 채점하거나 후보를 고르는 데 쓰지 않았다. 이전 잠금 확인 모델의 학습306일7344행과 이번 최종학습400일9600행을 구분해야 한다.

## 바꾼 이유와 규정

카탈로그 C6.157·6.161~6.165의 DC4 조사에서 후반 구간의 기록 일차와 날씨가 나타내는 계절 순서가 어긋났고, 기록 일차의 외삽이 EC 오차를 키울 가능성을 찾았다. `day`만 학습 입력으로 만든 계절 지표 `season`으로 바꿨다. 실제 달력을 복원했다는 증명은 아니다.

학습1차(day<179)의 외기온도·습도·일사·풍속 24시간 벡터로 각 변수의 평균과 표준편차를 fit한다. 학습2차(day>=179)는 표준화 RMSE 거리≤.05의 1차 짝들의 평균 일차를 기준점으로 삼는다. 온실별 단조 등위회귀(PAV) 후 일차를 선형보간하고 경계는 고정한다. 기준점2개 미만이면 원래 DC4의 최근접 짝 fallback을 쓴다. **평가 날은 자기 온실·일차로 학습변환표를 보간할 뿐 평가 날씨를 사용하지 않는다.** 자기 온실 현재/이전 평가 입력만으로 원시 특징·자정 특징·현재까지 누적 특징을 만든다. 학습 특징은 평가 MASK에 허용된 실내/액추에이터10열만 쓴다. 학습 외기4열은 변환 fit용이며 평가 모델 입력이 아니다. 날짜/온실은 보간과 인과적 그룹 구분에 사용하고 모델에 온실 ID를 추가하지 않았다.

## 고정 모델

원시예측은 `0.8 × R3 평균 + 0.2 × TabPFN 평균`이다. R3=`0.6ET+0.3LGB+0.1MLP`, 시드7·101·2024 평균. ET600/leaf1, LGB Tweedie800/lr.03/31leaves/minchild40, MLP128·64/alpha.01/earlystop. TabPFN V2 CPU float32는 2000학습행을 시드1·2·3·4로 각각 비복원추출한 문맥4개, 문맥당 estimator4 평균이다. `day`를 제거하고 마지막에 `season`을 붙이는 순서로 ET/PFN38열, LGB/MLP14열을 쓴다. 원시혼합 후 현재예측과 당일현재까지예측평균을 .5/.5로 섞고 전체 학습 EC 범위로 clip한다. 직전정답 STATE 보정은 없다. PFN문맥은 R3시드 사이에 공유되므로 세 시드가 완전히 독립인 실험은 아니다.

## 검증 근거 (공식 리더보드 점수 아님)

공개 검증은 잠금40일 및 검증일±1일을 fit에서 제외한 기존22fold×3R3시드다. 저장 OOF를 이번 제작에서도 NumPy와 math.fsum 두 방법으로 재계산했다. 3시드 예측을 행별 평균한 뒤 RMSE를 계산한 값이다.

| 검증기 | 평가 행수 | day v2 | 계절 v2 |
|---|---|---|---|
{rows}

각 시드×5검증기15/15칸이 개선했다. DIAG10은 온실층화5일 블록80개, bootstrap20000회, RNG918의 사전 규칙으로 판정했다. 이번 DC4/E40 두 후보에 Bonferroni k=2를 적용하여 각 시드 p_worse<.0125 및97.5% ΔMSE CI상한<0을 요구했다. bootstrap은 이미 독립검산된 고정 기록을 사용하고 이번 제작에서 다시 후보를 시험하지 않았다.

{bootstrap_lines}

| DIAG10 구간(진단 전용) | 일수 / 행수 | day v2 | 계절 v2 |
|---|---|---|---|
{segment}

잠금40일960행은 예측을 먼저 저장한 뒤 정답을 한 번만 읽었다. day v2 RMSE {lock['day_rmse']:.9f} → 계절 v2 {lock['season_rmse']:.9f}, 차 {lock['delta_rmse']:.9f}. 동일306일7344행 fit의 비교다. 최초 math.fsum/NumPy 검산 통과 기록과 예측 파일 SHA256만 재확인했고 잠금 정답을 재채점하지 않았다. 독립 ET 재현의 최대차4.44e-16과 계절변환≤1e-10은 이전 판정 보고서의 출처 기록이다.

## 재현과 규정 검사

Python3.12 및 requirements.txt의 고정 버전이 필요하다. ZIP의 입력·코드·TabPFN체크포인트·DLL은 동봉되어 있고 모델이 외부자료를 조회하지 않는다. 새 폴더에 풀고 `python -B -u reproduce.py --output 새출력폴더`를 실행한다. 기존 출력은 덮어쓰지 않는다. 이 PC에서는 `EC_CORE_DEPENDENCY_PATH=C:/work/farmai/.analysis-tools/python`, `EC_DEPENDENCY_PATH=C:/work/farmai/.analysis-tools/extra`로 동일 설치 라이브러리를 지정할 수 있다. 다른 PC는 requirements.txt를 설치하면 된다. CPU 학습에 시간이 걸린다.

깨끗한 ZIP 추출본에서 실제 전체 재학습을 완료했다. 두 실행의 EC bit 동일·CSV bytes 동일, 최대차0이다. 학습400일→계절표 `training_day_to_season.csv`와 평가60일 계절표 `evaluation_day_to_season.csv`, `season_transform.json`도 bytes 동일하다. 클로드 원본 pivot+sklearn 등위회귀로 독립계산한 최종 계절값의 학습 최대차 {season['training_max_difference']:.3g}, 평가 최대차 {season['evaluation_max_difference']:.3g} (허용1e-9). 평가행 셔플, 두 온실별3절단점의 미래·다른 온실 평가 입력변조에서 대상 특징 불변을 확인했다. 각 R3모델 및 PFN4문맥의 첫8대상행은 전체 평가 배치와 대상만 예측한 결과가 bit 동일하다. 변조 검사는 특징 수준6경우와 모델 배치독립8행 검사로 나누어 기록했으며 모든1440행×모든변조 조합을 재학습한 시험은 아니다. 자세한 증거는 verification.json이다.

CSV SHA256: `{sha(orig/'submission_10.csv')}`.

## 남은 위험과 팀원 결합

실행·재현·행 정합성의 신뢰도는 높다. 새 평가일 일반화의 신뢰도는 중간이다. DC4는 DC3와 공개 검증 결과를 본 뒤 설계했고 공개 검증기를 반복 사용했다. 이전5회차 `day` 제거의 리더보드 악화와 현재 계절지표 검증 이득 사이의 모순이 남는다(C6.165). 계절지표가 그 차이를 설명한다고 단정하지 않는다. 잠금40일도 같은 원자료에서 뽑은 표본이다. 전체400일 최종fit의 성능은 별도 미지 정답으로 측정하지 않았다. 리더보드 점수로 비중·시드·계절표를 선택하지 않았다. 새로운 공식 점수는 아직 없다.

팀원이 row_id로 온도를 채운 뒤 최종 CSV의 1440행/순서/두열 finite와 재현 ZIP의 합본 재현을 다시 확인해야 한다. 이 ZIP은 온도 빈칸인 EC 전달본을 그대로 재현한다. 팀원 온도 모델·최종합본은 이 패키지에 포함되지 않는다. 클로드 독립검산에는 CSV, 평가60일계절표, fresh_public_verification.json, verification.json을 전달하면 된다. 3회차 EC 대비 달력 이른/늦은날 변화는 별도 진단이며 후보선택에는 쓰지 않는다.
'''
    docpath=OUT/'설명자료_10_제출8회차.md';docpath.write_text(doc,encoding='utf-8')
    for n in ['submission_10.csv','training_day_to_season.csv','evaluation_day_to_season.csv','season_transform.json','manifest.json']:shutil.copyfile(orig/n,STAGE/n)
    shutil.copyfile(orig/'submission_10.csv',OUT/'submission_10.csv')
    for n in ['verification.json','fresh_public_verification.json',docpath.name]:shutil.copyfile(OUT/n,STAGE/n)
    zip_path=OUT/'팜모니_정형데이터_재현패키지_10.zip';archive(zip_path,STAGE)
    extract=OUT/'final_zip_check'
    with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None;z.extractall(extract)
    assert all(sha(extract/p.relative_to(STAGE))==sha(p) for p in STAGE.rglob('*') if p.is_file())
    final={'status':'PASS','csv':str(OUT/'submission_10.csv'),'zip':str(zip_path),'explanation':str(docpath),'temperature':'BLANK_FOR_TEAMMATE','final_zip_matches_code_and_inputs_actually_retrained':True,'files_sha256':{p.name:sha(p) for p in [OUT/'submission_10.csv',zip_path,docpath]},'verification':verification,'platform_submission':False}
    save(OUT/'delivery_manifest.json',final)
    save(HERE/'delivery_verification_v1.json',final)
    print('DELIVERY COMPLETE: EC season-v2; temperature blank; clean retraining byte-equal.',flush=True)
def main():
    prepare();run(STAGE,OUT/'source')
    archive(OUT/'clean_check_input.zip',STAGE)
    folder=OUT/'clean_check_extract'
    with zipfile.ZipFile(OUT/'clean_check_input.zip') as z:assert z.testzip() is None;z.extractall(folder)
    run(folder,folder/'replay');finalize()
if __name__=='__main__':main()
