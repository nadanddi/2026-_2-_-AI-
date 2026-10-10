import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
import json,torch,torchvision,numpy,PIL
modules={'torch':torch,'torchvision':torchvision,'numpy':numpy,'PIL':PIL}
paths={k:v.__file__ for k,v in modules.items()}
assert all('image_runtime_20261010_v1' in p and Path(p).exists() for p in paths.values())
result={'launcher':'run_image_runtime_v2.ps1','module_paths_owned':True,'module_paths':paths,'versions':{k:v.__version__ for k,v in modules.items()},'ROOT_matches_env':str(ROOT)==env.ROOT,'scope':'isolated preload plus repo bootstrap launcher probe'}
(Path(__file__).parent/'runtime_launcher_probe_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
