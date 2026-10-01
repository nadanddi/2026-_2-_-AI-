"""Runtime-only revision: preserve earlier artifacts, pin NumPy to CV version."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import argparse,shutil,json,zipfile,hashlib,csv,math
import numpy as np
import pandas as pd
LOCAL=ROOT/'집/코덱스/local/ec_v2_state_20261001_v1';OLD=LOCAL/'artifact';OUT=LOCAL/'artifact_numpy253';STAGE=OUT/'stage';CANON=OLD/'reproduction_check/replay'
CSV='ec_v2_state_v1_numpy253.csv';DOC='설명자료_ec_v2_state_v1_numpy253.md';ZIP='EC_v2_state_v1_numpy253_재현패키지.zip'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def makezip(path):
    assert not path.exists()
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(STAGE.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(STAGE).as_posix())
    with zipfile.ZipFile(path) as z:assert z.testzip() is None
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','verify'],required=True);a=p.parse_args()
    if a.mode=='prepare':
        assert not OUT.exists();STAGE.mkdir(parents=True)
        for folder in ['data','dlls']:shutil.copytree(OLD/'stage'/folder,STAGE/folder)
        for name in ['ec_model.py','env_extra.py','requirements.txt','merge_ec.py','locked_days.json','tabpfn-v2-regressor.ckpt','PROTOCOL.md','cv_verified.json','verified_scores.csv','verified_seed_scores.csv','verified_segments.csv']:shutil.copyfile(OLD/'stage'/name,STAGE/name)
        bootstrap=(OLD/'stage/env.py').read_text(encoding='utf-8').replace("extra=os.environ.get('EC_DEPENDENCY_PATH')", "core=os.environ.get('EC_CORE_DEPENDENCY_PATH')\nif core and core not in sys.path:sys.path.insert(0,core)\nextra=os.environ.get('EC_DEPENDENCY_PATH')")
        (STAGE/'env.py').write_text(bootstrap,encoding='utf-8')
        wrapper=(OLD/'stage/reproduce.py').read_text(encoding='utf-8').replace('import env_extra','import env_extra\nimport numpy as np\nassert np.__version__==\'2.5.3\', \'Use NumPy 2.5.3 from requirements.txt or EC_CORE_DEPENDENCY_PATH\'')
        (STAGE/'reproduce.py').write_text(wrapper,encoding='utf-8')
        manifest=json.loads((CANON/'manifest.json').read_text(encoding='utf-8'));assert manifest['versions']['numpy']=='2.5.3' and manifest['code_sha256']==sha(STAGE/'ec_model.py')
        shutil.copyfile(CANON/'ec_v2_state_v1.csv',OUT/CSV);shutil.copyfile(OUT/CSV,STAGE/CSV)
        for name in ['manifest.json','past_state_audit.csv']:shutil.copyfile(CANON/name,STAGE/name)
        doc=(OLD/'설명자료_ec_v2_state_v1.md').read_text(encoding='utf-8').replace('`ec_v2_state_v1.csv`','`ec_v2_state_v1_numpy253.csv`').replace('--ec ec_v2_state_v1.csv','--ec ec_v2_state_v1_numpy253.csv')
        doc+='''\n\n## 최종 전달본의 실행 환경 고정\n\n모델 구조는 v1과 동일하고 전달본은 v1_numpy253이다. 최초 독립 환경 NumPy2.3.5 결과와 검증 환경2.5.3 결과의 최대 예측 차이 .002620989는 허용치1e−8을 초과했다. 최초 전달 준비본은 재현 실패로 사용하지 않는다. 이 전달 CSV는 NumPy2.5.3으로 다시 생성한 결과이며, 재현기에도2.5.3 검사조건을 넣었다.\n\n기존 프로젝트 라이브러리를 쓸 경우 `$env:EC_CORE_DEPENDENCY_PATH='C:\\work\\farmai\\.analysis-tools\\python'`과 `$env:EC_DEPENDENCY_PATH='C:\\work\\farmai\\.analysis-tools\\extra'`를 설정한다. 다른 환경은 requirements.txt 버전을 설치한다. `python reproduce.py --output 새출력폴더`는 내부 모델명 그대로 ec_v2_state_v1.csv를 생성하며, 내용이 최종 전달 파일 ec_v2_state_v1_numpy253.csv와 일치하는지 동봉 verification.json으로 확인한다. 플랫폼 제출하지 않았고 개선 검증 실패 상태는 바뀌지 않았다.\n'''
        (OUT/DOC).write_text(doc,encoding='utf-8');shutil.copyfile(OUT/DOC,STAGE/DOC)
        note={'revision':'v1_numpy253','model_recipe_changed':False,'initial_numpy':'2.3.5','pinned_numpy':'2.5.3','initial_cross_version_maxdiff':.0026209890842436856,'initial_reproducibility_check':'FAILED','tolerance_not_relaxed':1e-8,'candidate_adopted':False}
        (STAGE/'runtime_revision.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
        temp=OUT/'clean_check_input.zip';makezip(temp);extract=OUT/'reproduction_check'
        with zipfile.ZipFile(temp) as z:
            for name in z.namelist():assert '..' not in Path(name).parts and not Path(name).is_absolute()
            z.extractall(extract)
        print(extract)
    else:
        a=pd.read_csv(OUT/CSV);b=pd.read_csv(OUT/'reproduction_check/replay/ec_v2_state_v1.csv');ids=pd.read_csv(STAGE/'data/sample_submission.csv')
        assert list(a.columns)==list(b.columns)==['row_id','sub_ec'] and len(a)==1440 and a.row_id.is_unique
        assert a.row_id.tolist()==b.row_id.tolist()==ids.row_id.tolist() and np.isfinite(a.sub_ec).all() and a.sub_ec.ge(0).all()
        diff=float(np.max(np.abs(a.sub_ec-b.sub_ec)));assert diff<=1e-8
        orig=json.loads((STAGE/'manifest.json').read_text(encoding='utf-8'));replay=json.loads((OUT/'reproduction_check/replay/manifest.json').read_text(encoding='utf-8'));assert orig['versions']==replay['versions'] and orig['inputs_sha256']==replay['inputs_sha256']
        audit=pd.read_csv(STAGE/'past_state_audit.csv');assert audit.state_source_day.notna().all() and (audit.state_source_day<audit.day).all() and audit.prev_public_gap_hours.gt(0).all()
        with (OUT/CSV).open(newline='',encoding='utf-8-sig') as f:v=[float(r['sub_ec']) for r in csv.DictReader(f)]
        with (OUT/'reproduction_check/replay/ec_v2_state_v1.csv').open(newline='',encoding='utf-8-sig') as f:w=[float(r['sub_ec']) for r in csv.DictReader(f)]
        assert len(v)==1440 and math.isclose(math.fsum(v)/len(v),float(a.sub_ec.mean()),abs_tol=1e-12) and min(v)==float(a.sub_ec.min()) and math.isclose(diff,max(abs(x-y) for x,y in zip(v,w)),abs_tol=1e-12)
        for p in STAGE.rglob('*'):
            if p.is_file():assert sha(p)==sha(OUT/'reproduction_check'/p.relative_to(STAGE))
        result={'status':'PASS','rows':1440,'columns':['row_id','sub_ec'],'sample_order':'PASS','finite_nonnegative':'PASS','prior_state_causality':'PASS','clean_extraction_full_retraining':'PASS','same_library_versions':'PASS','max_absolute_prediction_difference':diff,'independent_csv_arithmetic':'PASS','source_unchanged':'PASS','original_sha256':sha(OUT/CSV),'replay_sha256':sha(OUT/'reproduction_check/replay/ec_v2_state_v1.csv'),'numpy':'2.5.3','candidate_adopted':False,'platform_submitted':False}
        (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copyfile(OUT/'verification.json',STAGE/'verification.json');makezip(OUT/ZIP)
        (OUT/'delivery_manifest.json').write_text(json.dumps({'status':'PASS','revision':'v1_numpy253','files_sha256':{n:sha(OUT/n) for n in [CSV,DOC,ZIP,'verification.json']},'candidate_adopted':False,'platform_submitted':False},ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
