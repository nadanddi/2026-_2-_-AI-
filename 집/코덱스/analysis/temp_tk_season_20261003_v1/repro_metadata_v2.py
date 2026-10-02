from world import *
import os
from importlib.metadata import version
sys.path.insert(0,str(ROOT/'.analysis-tools/extra_gpu'));sys.path.insert(1,str(ROOT/'.analysis-tools/extra'))
import env_extra_gpu,torch
checkpoint=Path(os.environ['APPDATA'])/'tabpfn/tabpfn-v2-regressor.ckpt'
answer=dict(versions={n:version(n) for n in ['numpy','pandas','scipy','scikit-learn','lightgbm','tabpfn','torch']},checkpoint_file=checkpoint.name,checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,device=torch.cuda.get_device_name(0),precommit='b730e80',raw_hashes={str(p.relative_to(ROOT)):sha(p) for p in Path(env.DATA).glob('*.csv')},baseline_cache_hashes={p.name:sha(p) for p in [Path(env.LOCAL)/'temp_mask_v1_oof.npz']+[Path(env.LOCAL)/f'web_tabpfn_{v}_temp_{s}.npy' for s in ['DIAG10','EXT10','EXT12'] for v in ['v2','v6']]})
(HERE/'repro_metadata.json').write_text(json.dumps(answer,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(answer['versions']))
