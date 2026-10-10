import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,hashlib,io,itertools
import numpy as np
from PIL import Image,ImageOps
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local/image_source_mapping_20261010_v1'
q=json.loads((RUN/'qa_v1.json').read_text(encoding='utf-8'));fixed=json.loads((RUN/'selection_v1.json').read_text(encoding='utf-8'))['rows']
assert [(r['source_archive'],r['source_member']) for r in q['selected']]==[(r['source_archive'],r['source_member']) for r in fixed]
checks=[];reads=0;pixels=0;hashes=[]
for r in q['selected']:
    with (LOCAL/r['source_archive']).open('rb') as f:f.seek(r['offset']);raw=f.read(r['bytes'])
    reads+=len(raw);assert hashlib.sha256(raw).hexdigest()==r['payload_sha256']
    with Image.open(io.BytesIO(raw)) as im:
        assert list(im.size)==r['actual_size'];pixels+=im.width*im.height
        im.load();gray=np.asarray(ImageOps.exif_transpose(im).convert('RGB').convert('L').resize((9,8),Image.Resampling.LANCZOS))
    bits=(gray[:,:-1]>gray[:,1:]).reshape(-1);computed=int.from_bytes(np.packbits(bits).tobytes(),'big')
    assert computed==int(r['dhash'],16);hashes.append(bits)
    assert hashlib.sha256(Path(r['derived_path']).read_bytes()).hexdigest()==r['derived_sha256']
    checks.append({'image_id':r['image_id'],'source_sha_decode_size_dhash_derived_sha_match':True})
near=[]
for i,j in itertools.combinations(range(len(hashes)),2):
    d=int(np.count_nonzero(hashes[i]!=hashes[j]))
    if d<=6:near.append((q['selected'][i]['source_member'],q['selected'][j]['source_member'],d))
assert near==[(r['a'],r['b'],r['distance']) for r in q['near_candidates']]
assert pixels==q['decoded_pixels']
result={'selected_rows':len(checks),'all_checks_pass':True,'pixel_sum':pixels,'source_payload_read_bytes':reads,'mismatched_dimensions':sum(r['actual_size']!=[int(r['width']),int(r['height'])] for r in q['selected']),'near_candidate_pairs':len(near),'independent_method':'NumPy packbits and boolean Hamming distances against original Python bit-shift/popcount; source payloads freshly decoded','scope':'64 fixed development images only; no full-data uniqueness or score claim'}
with (RUN/'qa_verification_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
