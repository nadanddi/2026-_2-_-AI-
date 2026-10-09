"""Execute immutable round-9 temperature script from a fresh ZIP extraction."""
from pathlib import Path
import sys,os,json,hashlib,runpy,platform,time
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
P=json.loads((H/'preparation_v1.json').read_text(encoding='utf-8'))
CODE=Path(P['temp_code']);TEMP=CODE.parent;OUT=Path(P['output'])
assert not OUT.exists()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P['zip'])==P['zip_sha']
for n,v in P['temp_package_files'].items():assert sha(TEMP/n)==v,(n,'package changed')
for n,v in P['data_sha'].items():assert sha(TEMP/'data'/n)==v
assert sha(Path(P['cache'])/'tabpfn-v2-regressor.ckpt')==P['checkpoint_sha']
for n in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_DISABLE_TELEMETRY','HF_HUB_DISABLE_TELEMETRY']:os.environ[n]='1'
os.environ['TABPFN_MODEL_CACHE_DIR']=P['cache']
os.environ['AGRI_DATA']=str(TEMP/'data')
sys.path.insert(0,str(ROOT/'.analysis-tools/python'));sys.path.append(str(ROOT/'.analysis-tools/extra'))
handles=[]
for d in [ROOT/'.analysis-tools/msvc',ROOT/'.analysis-tools/extra/torch/lib',Path(P['package'])/'EC/dlls']:
    if d.exists() and hasattr(os,'add_dll_directory'):handles.append(os.add_dll_directory(str(d)))
sys.path.insert(0,str(CODE));os.chdir(CODE)
import env
assert Path(env.__file__).resolve()==(CODE/'env.py').resolve()
assert Path(env.common.__file__).resolve()==(CODE/'common.py').resolve()
assert Path(env.common.DATA).resolve()==(TEMP/'data').resolve()
import numpy,pandas,scipy,sklearn,lightgbm,torch,tabpfn
from tabpfn.model_loading import get_cache_dir
assert get_cache_dir().resolve()==Path(P['cache']).resolve()
modules=[numpy,pandas,scipy,sklearn,lightgbm,torch,tabpfn]
versions={m.__name__:m.__version__ for m in modules}
assert versions==dict(numpy='2.5.3',pandas='3.0.1',scipy='1.18.1',sklearn='1.9.1',lightgbm='4.7.0',torch='2.14.0+cpu',tabpfn='9.0.0')
script=CODE/'make_submission_v13_temp.py'
receipt={'status':'RUNNING_ORIGINAL_PACKAGED_TEMPERATURE_SCRIPT','pid':os.getpid(),'python':platform.python_version(),'python_exe':sys.executable,'versions':versions,'module_paths':{m.__name__:m.__file__ for m in [env,env.common]+modules},'source_sha':sha(script),'wrapper_sha':sha(__file__),'checkpoint_sha':P['checkpoint_sha'],'cache':str(get_cache_dir()),'torch_threads':torch.get_num_threads(),'source_changed':False,'output':str(OUT),'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'new_submission':False}
with (H/'runtime_v1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print('RUNTIME_READY',json.dumps(receipt,ensure_ascii=False),flush=True)
sys.argv=[str(script),str(OUT)]
start=time.monotonic()
runpy.run_path(str(script),run_name='__main__')
receipt.update(status='EXECUTION_COMPLETED_COMPARE_PENDING',seconds=time.monotonic()-start,output_files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},project_module_paths={n:str(m.__file__) for n,m in sys.modules.items() if getattr(m,'__file__',None) and n in ['env','common','harness','resid_reset_features','temp_mask_v1','train_flags_v6','make_submission_v3','make_submission_v7']})
for n,p in receipt['project_module_paths'].items():assert Path(p).resolve().is_relative_to(CODE.resolve()),(n,p)
for n,v in P['temp_package_files'].items():assert sha(TEMP/n)==v,(n,'package changed after run')
with (H/'execution_v1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print('EXECUTION_COMPLETED',receipt['seconds'],flush=True)
