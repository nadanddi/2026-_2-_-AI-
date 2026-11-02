import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent; L=Path.home()/'storage/nadan/farmai_img/data/local'
rr=json.loads((HERE/'miss_queries_v1.json').read_text()); out=[]
for r in rr:
    p=L/r['dataset']/'mask'/Path(r['file']).with_suffix('.png').name
    v=dict(r,area=None,mean_alpha=None,reason='부분 유형 아님')
    if r['kind'].startswith('partial') or r['kind']=='pstyle_color':
        v['reason']='저장 마스크 없음'
        if p.is_file():
            with Image.open(p) as im: a=np.asarray(im.convert('L'))
            v.update(area=float((a>127.5).mean()),mean_alpha=float(a.mean()/255),reason='저장 마스크상 면적')
    out.append(v)
with (HERE/'miss_masks_v1.json').open('x',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=2)
print('MASKS_DONE',len(out),'available',sum(v['area'] is not None for v in out))
