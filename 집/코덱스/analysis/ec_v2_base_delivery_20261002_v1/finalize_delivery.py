from pathlib import Path
import sys,csv,json,hashlib,zipfile,shutil,math
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=ROOT/'집/코덱스/local/ec_v2_base_delivery_20261002_v1/artifact'
STAGE=OUT/'stage'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    target=OUT/'ec_v2_base_v1.csv';replay=OUT/'clean_check_extract/replay'
    orig=read_csv(target);actual=read_csv(replay/'ec_v2_base_v1.csv')
    assert len(orig)==len(actual)==1440 and [r['row_id'] for r in orig]==[r['row_id'] for r in actual]
    a=np.array([float(r['sub_ec']) for r in orig]);b=np.array([float(r['sub_ec']) for r in actual])
    maxgap=float(np.max(np.abs(a-b)));assert maxgap<=1e-8
    prepare=json.loads((OUT/'preparation_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(STAGE/p)==h for p,h in prepare['files_sha256'].items())
    assert all(sha(OUT/'clean_check_extract'/p)==h for p,h in prepare['files_sha256'].items())
    assert math.isclose(math.fsum(b)/1440,float(b.mean()),abs_tol=1e-15)
    report={'status':'PASS','rows':1440,'id_order_equal':True,'max_abs_prediction_difference':maxgap,
            'prediction_bits_equal':bool(np.array_equal(a.view(np.uint64),b.view(np.uint64))),
            'csv_bytes_equal':sha(target)==sha(replay/'ec_v2_base_v1.csv'),
            'csv_sha256':sha(target),'reproduced_csv_sha256':sha(replay/'ec_v2_base_v1.csv'),
            'training_rows':7344,'training_days':306,'final_lock_scored':False,'state_correction':0,
            'actual_retraining_from_clean_extract':True,'platform_submission':False,
            'mean':float(b.mean()),'minimum':float(b.min()),'maximum':float(b.max()),
            'stage_files_before_verification':prepare['files_sha256']}
    assert not (OUT/'verification.json').exists()
    (OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copyfile(OUT/'verification.json',STAGE/'verification.json')
    doc=f'''# EC v2 전달본 v1 · 2026-10-02 집 코덱스

기존 v2의 EC 전용 전달본이다. 팀원 온도 결과와 row_id로 합칠 수 있다. 이번 제거4안/CatBoost 실험은모두기각되어모델은기존v2를유지했다. 실제 플랫폼 제출은 하지 않았다.

- CSV: ec_v2_base_v1.csv, row_id/sub_ec, 1440행, sample순서.
- 원시R3(.6ET+.3LGB+.1MLP) 시드7/101/2024 평균80% + TabPFN V2 context1..4 평균20%. ET/PFN FULL38, LGB/MLP BASE14. 현재예측/당일현재까지평균 .5/.5, fold학습범위clip. 직전정답STATE보정0.
- 기존검증 DIAG10 .213874/A .230302/B .218769/EXT10 .414091/EXT12 .416128 (3시드평균, 공식점수아님). 기존R3보다후반이약한점과검증기반복사용의불확실성은유지된다. 7회차state점수 .2774와다른모델이다. 새로운공식성능개선을주장하지않는다.
- 학습7344행306일. 잠금40일을숫자변환전제외하고±1일purge. package train_y는온도/잠금정답없음. 입력·checkpoint는동봉, 외부조회없이CPU실행.

## 재현

Python3.12와requirements.txt 고정버전을설치한다. 이PC에서는 EC_CORE_DEPENDENCY_PATH=C:/work/farmai/.analysis-tools/python, EC_DEPENDENCY_PATH=C:/work/farmai/.analysis-tools/extra를지정해같은라이브러리를쓴다. ZIP을새폴더에풀고 `python -B -u reproduce.py --output 새출력폴더`를실행하면EC CSV와학습manifest가생성된다. 기존출력을덮어쓰지않는다. 큰CPU모델이라시간이걸린다.

깨끗한추출본에서실제재학습검산을완료했다. 기준캐시대비최대차 {maxgap:.3g} (사전허용1e-8), bit동일 {report['prediction_bits_equal']}, CSV바이트동일 {report['csv_bytes_equal']}. verification.json에숫자/해시가있다. CSV SHA256: {sha(target)}.

## 팀원 온도와 합치기

동봉 merge_ec.py는기존온도CSV의row_id/sub_temp를보존하고EC만붙인다. `python merge_ec.py --temperature 온도CSV --ec ec_v2_base_v1.csv --output 새합본CSV`. 실제온도모델·합본CSV·플랫폼제출은이전달본에서만들지않았다.
'''
    docpath=OUT/'설명자료_ec_v2_base_v1.md';assert not docpath.exists()
    docpath.write_text(doc,encoding='utf-8');shutil.copyfile(docpath,STAGE/docpath.name)
    archive=OUT/'EC_v2_base_v1_재현패키지.zip';assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(STAGE.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(STAGE).as_posix())
    extracted=OUT/'final_zip_check';assert not extracted.exists()
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None;z.extractall(extracted)
    assert all(sha(extracted/p)==h for p,h in prepare['files_sha256'].items())
    assert sha(extracted/'verification.json')==sha(OUT/'verification.json')
    final={'status':'PASS','csv':str(target),'zip':str(archive),'explanation':str(docpath),
           'files_sha256':{p.name:sha(p) for p in [target,archive,docpath]},'verification':report,
           'final_zip_verified_matches_actual_clean_training_inputs_and_code':True,'platform_submission':False}
    (OUT/'delivery_manifest.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in final.items() if k!='verification'},ensure_ascii=False))
if __name__=='__main__':main()
