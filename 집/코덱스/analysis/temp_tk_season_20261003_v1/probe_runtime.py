from pathlib import Path
import sys,os,json
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
sys.path.insert(0,str(ROOT/'.analysis-tools/extra_gpu'));sys.path.insert(1,str(ROOT/'.analysis-tools/extra'))
import env_extra_gpu
import numpy as np,torch,tabpfn
answer=dict(python=sys.version,torch=str(torch.__version__),torch_path=torch.__file__,cuda=torch.cuda.is_available(),device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,tabpfn_path=tabpfn.__file__,numpy=str(np.__version__))
(H/'runtime.json').write_text(json.dumps(answer,indent=2),encoding='utf-8');print(json.dumps(answer))
