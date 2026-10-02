from support import *
import platform,sklearn,lightgbm,scipy
sys.path.insert(0,str(ROOT/'.analysis-tools/extra_gpu'));sys.path.insert(1,str(ROOT/'.analysis-tools/extra'))
import env_extra_gpu,torch,tabpfn
sources=list((ROOT/'집/코덱스/analysis/temp_season_20261002_v1/reference').glob('*.py'))+[ROOT/'집/코덱스/analysis/codex_independent/rl_ec_v1/run.py',ROOT/'집/코덱스/analysis/ec_submission10_season_20261002_v1/season.py',ROOT/'집/클로드/research/env.py']
savej(HERE/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,lightgbm=lightgbm.__version__,torch=str(torch.__version__),tabpfn_path=str(tabpfn.__file__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),pfn_checkpoint_sha256=sha(Path.home()/'AppData/Roaming/tabpfn/tabpfn-v2-regressor.ckpt'),external_source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}))
print('ENVIRONMENT_RECORDED',flush=True)
