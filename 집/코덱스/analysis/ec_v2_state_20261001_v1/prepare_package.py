from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import csv,json,shutil,hashlib
HERE=Path(__file__).resolve().parent;OUT=ROOT/'집/코덱스/local/ec_v2_state_20261001_v1/artifact';STAGE=OUT/'stage'
assert not STAGE.exists(),'Use a new stage';(STAGE/'data').mkdir(parents=True)
locks={(z['farm'],int(z['day'])) for z in json.loads((Path(env.CODEX)/'ec_final_lock/locked_days.json').read_text(encoding='utf-8'))['selected']}
raw=['in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
counts={}
for name,cols in [('train_X.csv',['row_id']+raw),('test_X.csv',['row_id']+raw),('train_y.csv',['row_id','sub_ec']),('sample_submission.csv',['row_id'])]:
    n=0
    with (Path(env.DATA)/name).open(newline='',encoding='utf-8-sig') as fi,(STAGE/'data'/name).open('w',newline='',encoding='utf-8-sig') as fo:
        w=csv.DictWriter(fo,fieldnames=cols);w.writeheader()
        for r in csv.DictReader(fi):
            farm,day,_=r['row_id'].split('_')
            if farm not in ['F13','F47']:continue
            if name=='train_y.csv' and (farm,int(day)) in locks:continue
            w.writerow({c:r[c] for c in cols});n+=1
    counts[name]=n
assert counts=={'train_X.csv':9600,'test_X.csv':1440,'train_y.csv':8640,'sample_submission.csv':1440}
for source,dest in [(HERE/'ec_model.py','ec_model.py'),(HERE/'PROTOCOL.md','PROTOCOL.md'),(Path(env.CODEX)/'ec_final_lock/locked_days.json','locked_days.json'),(Path.home()/'AppData/Roaming/tabpfn/tabpfn-v2-regressor.ckpt','tabpfn-v2-regressor.ckpt')]:shutil.copyfile(source,STAGE/dest)
(STAGE/'dlls').mkdir()
dlls={}
for p in (ROOT/'.analysis-tools/msvc').rglob('*.dll'):dlls.setdefault(p.name,p)
for name,p in dlls.items():shutil.copyfile(p,STAGE/'dlls'/name)
bootstrap='''from pathlib import Path
import os,sys,glob
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
_handles=[]
extra=os.environ.get('EC_DEPENDENCY_PATH')
if extra and extra not in sys.path:sys.path.append(extra)
dirs=[str(ROOT/'dlls')]
for p in sys.path:
    if p:
        dirs.extend(glob.glob(str(Path(p)/'*'/'*.libs')))
        dirs.extend(glob.glob(str(Path(p)/'*.libs')))
        dirs.extend(glob.glob(str(Path(p)/'torch'/'lib')))
if hasattr(os,'add_dll_directory'):
    for p in dirs:
        if Path(p).is_dir():
            try:_handles.append(os.add_dll_directory(p))
            except OSError:pass
'''
(STAGE/'env.py').write_text(bootstrap,encoding='utf-8');(STAGE/'env_extra.py').write_text('import env\n',encoding='utf-8')
wrapper='''from pathlib import Path
import sys,runpy
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import env
import env_extra
assert Path(env.__file__).resolve().parent==HERE
for option,path in [('--data',HERE/'data'),('--lock',HERE/'locked_days.json'),('--checkpoint',HERE/'tabpfn-v2-regressor.ckpt')]:
    if option not in sys.argv:sys.argv.extend([option,str(path)])
runpy.run_path(str(HERE/'ec_model.py'),run_name='__main__')
'''
(STAGE/'reproduce.py').write_text(wrapper,encoding='utf-8')
(STAGE/'merge_ec.py').write_text('''import argparse
from pathlib import Path
import numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--temperature',type=Path,required=True);p.add_argument('--ec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists()
t=pd.read_csv(a.temperature);e=pd.read_csv(a.ec)
assert 'row_id' in t and 'sub_temp' in t and list(e.columns)==['row_id','sub_ec']
assert t.row_id.is_unique and e.row_id.is_unique and len(t)==len(e)==1440 and set(t.row_id)==set(e.row_id)
q=t[['row_id','sub_temp']].merge(e,on='row_id',how='left',validate='one_to_one',sort=False)
assert q.row_id.tolist()==t.row_id.tolist() and np.isfinite(q[['sub_temp','sub_ec']].to_numpy()).all()
q.to_csv(a.output,index=False,float_format='%.17g',encoding='utf-8-sig')
print(a.output)
''',encoding='utf-8')
(STAGE/'requirements.txt').write_text('numpy==2.5.3\npandas==3.0.1\nscikit-learn==1.9.1\nlightgbm==4.7.0\ntorch==2.14.0\ntabpfn==9.0.0\nthreadpoolctl\n',encoding='utf-8')
manifest={'status':'PASS','data_counts':counts,'temperature_labels_included':False,'locked_labels_included':False,'files_sha256':{str(p.relative_to(STAGE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in STAGE.rglob('*') if p.is_file()}}
(OUT/'stage_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'stage':str(STAGE),'counts':counts},ensure_ascii=False,indent=2))
