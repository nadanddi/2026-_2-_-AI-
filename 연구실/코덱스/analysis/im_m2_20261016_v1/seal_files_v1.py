"""Hash R5 and other gate-image inputs; read no score arrays or performance."""
import sys,json,hashlib,csv
from pathlib import Path
HERE=Path(__file__).resolve().parent
B=Path.home()/'storage/nadan/farmai_img'; L=B/'data/local'
pub=Path.home()/'storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플'
paths=list((L/'im_set_r5/img').glob('*.jpg'))+[L/'im_set_r5/manifest.csv',L/'feats_h/im_set_r5.npz']
names=('im_set_h3','im_set_h3j','im_set_h3s','im_set_h3g','im_set_gptr','im_set_gptr_src',
       'im_set_newreal','im_set_realg','im_set_realn','im_set_r2','im_set_r3','im_set_r4',
       'im_set_gpt3','im_set_gpt1','im_set_sd2','im_set_r3p_sharp','im_set_r3p_area',
       'im_set_r3p_png','im_set_r3p_webp','im_set_r3p_c444')
for sn in names: paths.extend((L/sn/'img').glob('*.jpg'))
for sn in ('real','brown','bright','qwen','v7'): paths.extend((L/'sealed_c3'/sn/'img').glob('*.jpg'))
paths.extend((L/'preds/_jig_hold').glob('j*.jpg'))
paths.append(pub/'sample_labels.csv')
with (pub/'sample_labels.csv').open() as f: paths.extend(pub/'images'/r['file'] for r in csv.DictReader(f))
paths.extend(HERE/n for n in ('im66_m2_v1.py','run_m2_v2.py','run_m2_v3.py','r5_integrity_v1.json','preflight/state_v1.json','verification_preflight_v1.json'))
record={}
for p in sorted(set(paths)):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    record[str(p)]=h.hexdigest()
with (HERE/'sealed_files_v1.json').open('x') as f: json.dump(record,f,indent=2)
print('SEALED_FILES_HASHED',len(record),'no score arrays read')
