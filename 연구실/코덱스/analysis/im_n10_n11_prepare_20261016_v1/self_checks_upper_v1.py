import sys,json,zipfile,hashlib,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;B=Path.home()/'storage/nadan';OFF=B/'공용/대회자료/이미지데이터/참가자_배포'
result={}
for n in (10,):
    root=HERE/f'build_upper_v1/n{n}';arc=root/f'팜모니_나단{n}.zip';clean=root/'clean_check_v1';clean.mkdir(exist_ok=False)
    with zipfile.ZipFile(arc) as z:
        assert all((clean/k).resolve().is_relative_to(clean.resolve()) for k in z.namelist());assert z.testzip() is None;z.extractall(clean)
    team=clean/f'팜모니_나단{n}'
    proc=subprocess.run([sys.executable,'-B',str(OFF/'self_check.py'),'--team_dir',str(team),'--images_dir',str(OFF/'공개샘플/images'),'--labels_csv',str(OFF/'공개샘플/sample_labels.csv'),'--time_limit','300','--seed','161610'],cwd=clean,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (root/'self_check_v1.log').write_text(proc.stdout,encoding='utf-8');assert proc.returncode==0,(n,proc.stdout)
    r=json.loads((clean/'self_check_result.json').read_text(encoding='utf-8'));assert not r['errors'] and r['info']['sample_metrics']['accuracy']==1 and r['info']['run']['order_check']['consistent']
    (root/'self_check_result_v1.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');result[str(n)]=r
    print('OFFICIAL_SELF_CHECK_PASS',n,flush=True)
with (HERE/'self_checks_upper_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
