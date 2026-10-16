"""Verify the existing N9 ZIP in a fresh extraction without modifying it."""
import os,sys,json,hashlib,zipfile,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
B=Path.home()/'storage/nadan/farmai_img';SRC=B/'submit_img09/팜모니_나단9.zip'
EXPECTED='6c109d45e3281399b4cf42e3030f9ab66aa365dc51df7c19a1434b489c09d316'
assert hashlib.sha256(SRC.read_bytes()).hexdigest()==EXPECTED
D=HERE/'clean_extract_v1';D.mkdir(exist_ok=False)
with zipfile.ZipFile(SRC) as z:
    for n in z.namelist():
        target=(D/n).resolve();assert target.is_relative_to(D.resolve())
    assert z.testzip() is None
    z.extractall(D)
TEAM=D/'팜모니_나단9';assert (TEAM/'inference.py').is_file()
GPU=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
assert not GPU,'GPU0_BUSY_STOP'
(HERE/'gpu_before_check_v1.log').write_bytes(subprocess.check_output(['nvidia-smi','--id=0']))
P=B.parent/'공용/대회자료/이미지데이터/참가자_배포'
cmd=[sys.executable,'-B',str(P/'self_check.py'),'--team_dir',str(TEAM),'--images_dir',str(P/'공개샘플/images'),'--labels_csv',str(P/'공개샘플/sample_labels.csv'),'--time_limit','300','--seed','1616']
subprocess.run(cmd,cwd=HERE,check=True)
result=json.loads((HERE/'self_check_result.json').read_text(encoding='utf-8'));assert not result['errors']
assert hashlib.sha256(SRC.read_bytes()).hexdigest()==EXPECTED
out=dict(pass_all=True,zip_sha256=EXPECTED,zip_bytes=SRC.stat().st_size,original_unchanged=True,clean_extraction=True,official_self_check_pass=True,colab_t4_new_measurement=False)
with (HERE/'verification_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('N9_FALLBACK_CHECK_PASS',flush=True)
