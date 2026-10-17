"""Colab T4 timing only; no labels, tuning, uploads, or intermediate-stat reports."""
import argparse,json,subprocess,sys,tempfile,time,zipfile
from pathlib import Path
WORKER=r'''
import json,sys,time
from pathlib import Path
import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0),torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU 없음'
team,images,out=sys.argv[1:];paths=sorted(str(p) for p in Path(images).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
assert len(paths)>0
sys.path.insert(0,team)
import inference as I
t0=time.perf_counter();m=I.load_model(str(Path(team)/'model'));torch.cuda.synchronize();t1=time.perf_counter();p=I.predict(m,paths);torch.cuda.synchronize();t2=time.perf_counter()
assert len(p)==len(paths) and all(float('-inf')<float(v)<float('inf') and 0<=float(v)<=1 for v in p)
r=dict(gpu=torch.cuda.get_device_name(0),images=len(paths),load_s=t1-t0,predict_s=t2-t1,total_s=t2-t0,limit_300s=(t2-t0<=300),all_521_or_524_images=(len(paths) in (521,524)))
Path(out).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(r,ensure_ascii=False),flush=True)
'''
ap=argparse.ArgumentParser();ap.add_argument('--zips',nargs='+',required=True);ap.add_argument('--images',required=True);ap.add_argument('--out',default='T4_실측결과.json');a=ap.parse_args()
results={}
with tempfile.TemporaryDirectory(prefix='nadan_t4_') as tmp:
    w=Path(tmp);worker=w/'worker.py';worker.write_text(WORKER,encoding='utf-8')
    for i,name in enumerate(a.zips):
        dst=w/f'package{i}';dst.mkdir()
        with zipfile.ZipFile(name) as z:
            assert all((dst/k).resolve().is_relative_to(dst.resolve()) for k in z.namelist());z.extractall(dst)
        team=next(p.parent for p in dst.rglob('inference.py'));out=w/f'time{i}.json'
        subprocess.run([sys.executable,'-B',str(worker),str(team),a.images,str(out)],check=True)
        results[Path(name).name]=json.loads(out.read_text(encoding='utf-8'))
Path(a.out).write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print('실측결과 저장:',a.out)
