"""Hash R5 and other gate-image inputs; read no score arrays or performance."""
import sys,json,hashlib,csv
from pathlib import Path
HERE=Path(__file__).resolve().parent
B=Path.home()/'storage/nadan/farmai_img'; L=B/'data/local'
pub=Path.home()/'storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플'
def listed_images(base):
    with (base/'manifest.csv').open(encoding='utf-8-sig') as f:
        return [base/'img'/r['file'] for r in csv.DictReader(f)]
r5=listed_images(L/'im_set_r5'); assert len(r5)==700 and len(set(r5))==700
paths=r5+[L/'im_set_r5/manifest.csv',L/'feats_h/im_set_r5.npz']
names=('im_set_h3','im_set_h3j','im_set_h3s','im_set_h3g','im_set_gptr','im_set_gptr_src',
       'im_set_newreal','im_set_realg','im_set_realn','im_set_r2','im_set_r3','im_set_r4',
       'im_set_gpt3','im_set_gpt1','im_set_sd2','im_set_r3p_sharp','im_set_r3p_area',
       'im_set_r3p_png','im_set_r3p_webp','im_set_r3p_c444')
for sn in names: paths.extend(listed_images(L/sn))
for sn in ('real','brown','bright','qwen','v7'): paths.extend(listed_images(L/'sealed_c3'/sn))
paths.extend((L/'preds/_jig_hold').glob('j*.jpg'))
paths.append(pub/'sample_labels.csv')
with (pub/'sample_labels.csv').open() as f: paths.extend(pub/'images'/r['file'] for r in csv.DictReader(f))
paths.extend(HERE/n for n in ('im66_m2_v1.py','run_m2_v2.py','run_m2_v3.py','run_m2_remote_v3.sh','seal_files_v2.py','r5_integrity_v1.json','preflight/state_v1.json','verification_preflight_v1.json'))
record={}
for p in sorted(set(paths)):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    record[str(p)]=h.hexdigest()
with (HERE/'sealed_files_v1.json').open('x') as f: json.dump(record,f,indent=2)
with (HERE/'seal_coverage_v1.json').open('x') as f:
    json.dump(dict(r5_images=700,all_manifest_images_included=True,feature_manifest_public_labels_candidate_actual_runner_included=True,
                   hashed_files=len(record),seal_sha256=hashlib.sha256((HERE/'sealed_files_v1.json').read_bytes()).hexdigest(),
                   scope='Image inputs frozen at this seal timestamp; not certified back to preflight start.'),f,indent=2)
print('SEALED_FILES_HASHED',len(record),'no score arrays read')
