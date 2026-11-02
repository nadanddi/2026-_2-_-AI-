"""Read-only experiment input inventory and outcome hashes; no R5 access."""
import sys, platform, hashlib, json
from pathlib import Path
import numpy, sklearn, scipy, PIL
HERE=Path(__file__).resolve().parent
B=Path.home()/'storage/nadan/farmai_img'
L=B/'data/local'
sets=('im_set_h3','im_set_h3j','im_set_h3s','im_set_h3g','im_set_gptr','im_set_gptr_src',
      'im_set_r2','im_set_r3','im_set_r4','im_set_newreal','im_set_realg','im_set_realn',
      'im_set_r3p_sharp','im_set_r3p_area','im_set_r3p_png','im_set_r3p_webp','im_set_r3p_c444',
      'im_set_gpt3','im_set_gpt1','im_set_sd2')
def entry(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
inputs=[B/'data/image/im44_h_train_eval.py']+[L/'preds'/s for s in ('h_model_h2.npz','ip_model_v1.npz','ip_theta_v1.json')]
for sn in sets: inputs += [L/sn/'manifest.csv',L/'feats_h'/f'{sn}.npz']
for tag in ('real','brown','bright','qwen','v7'):
    inputs += [L/'sealed_c3'/tag/'manifest.csv',L/'feats_h'/f'sealed_{tag}.npz']
inputs += [L/'feats_h/jig500.npz',L/'feats_h/pub21.npz']
obj=dict(python=sys.version,platform=platform.platform(),packages={x.__name__:x.__version__ for x in (numpy,sklearn,scipy,PIL)},
         input_hashes=[entry(p) for p in inputs],outcome_hashes=[entry(p) for p in sorted(HERE.iterdir()) if p.is_file()])
with (HERE/'provenance_v1.json').open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False)
print('PROVENANCE_DONE',len(inputs),len(obj['outcome_hashes']))
