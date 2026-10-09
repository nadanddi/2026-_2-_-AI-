import sys,io,json,csv,hashlib,math,collections
from pathlib import Path
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
import numpy as np
from PIL import Image,ImageOps
from scipy.fft import dct
HERE=Path(__file__).resolve().parent
ZIP=Path('G:/내 드라이브/농업 AI 경진대회/이미지 미션/New_Sample.zip의 사본')
COMP=ROOT/'공용'/'대회자료'/'이미지데이터'/'참가자_배포'/'공개샘플'
rows=json.loads((HERE/'records_v1.json').read_text(encoding='utf-8'))
profile=json.loads((HERE/'profile_v1.json').read_text(encoding='utf-8'))
comparison=json.loads((HERE/'comparison_v1.json').read_text(encoding='utf-8'))
arrays=np.load(HERE/'distances_v1.npz')
a=[];b=[]; metas=[]; mismatches=[]
def independent(raw,expected):
 assert hashlib.sha256(raw).hexdigest()==expected['sha256']
 with Image.open(io.BytesIO(raw)) as im:
  im.load(); assert im.size==(expected['width'],expected['height']) and im.mode==expected['mode']
  native=hashlib.sha256(str((im.size,im.mode)).encode()+im.tobytes()).hexdigest()
  assert native==expected['native_pixel_sha256']
  rgb=ImageOps.exif_transpose(im).convert('RGB')
  pixel=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
  assert pixel==expected['pixel_sha256']
  g=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
  c=dct(dct(g,type=2,norm='ortho',axis=0),type=2,norm='ortho',axis=1)[:8,:8].ravel()
  # Independent analytic cosine transform check, avoiding scipy implementation.
  idx=np.arange(32); k=idx[:,None]
  transform=np.cos(math.pi*(2*idx+1)*k/64)*math.sqrt(2/32); transform[0,:]/=math.sqrt(2)
  manual=(transform@g@transform.T)[:8,:8].ravel()
  assert np.max(np.abs(c-manual))<1e-8
  med=sorted(c[1:])[31]; bits=[0]+[int(x>med) for x in c[1:]]
  assert bits==expected['phash_bits']
  small=np.array(rgb.resize((128,128),Image.Resampling.BILINEAR))
  counts=[]
  for channel in range(3):
   counts.extend(np.bincount((small[:,:,channel].ravel()//32).astype(int),minlength=8).tolist())
  hist=[n/sum(counts) for n in counts]
  assert np.max(np.abs(np.array(hist)-expected['histogram']))<1e-15
 return {'bits':bits,'hist':hist,'sha':hashlib.sha256(raw).hexdigest(),'pixel':pixel}
with ZipFile(ZIP) as z:
 jpgs=sorted(n for n in z.namelist() if n.lower().endswith('.jpg'))
 jsons=sorted(n for n in z.namelist() if n.lower().endswith('.json'))
 assert len(jpgs)==len(jsons)==len(rows['aihub'])==340
 for i,row in enumerate(rows['aihub']):
  a.append(independent(z.read(row['name']),row))
  key=Path(row['name']).stem
  matches=[n for n in jsons if Path(n).stem==key]; assert len(matches)==1
  meta=json.loads(z.read(matches[0]).decode('utf-8-sig'))['images']; assert meta==row['meta']; metas.append(meta)
  if (int(meta['width']),int(meta['height']))!=(row['width'],row['height']): mismatches.append({'name':row['name'],'json_width':int(meta['width']),'json_height':int(meta['height']),'actual_width':row['width'],'actual_height':row['height']})
  if (i+1)%50==0: print('VERIFIED_AI',i+1,flush=True)
labels=list(csv.DictReader((COMP/'sample_labels.csv').open(encoding='utf-8-sig')))
assert len(labels)==21 and sum(int(r['label'])==0 for r in labels)==3 and sum(int(r['label'])==1 for r in labels)==18
for row in rows['competition']: b.append(independent((COMP/'images'/row['name']).read_bytes(),row))
manual_ham=[]; manual_color=[]
for q in b:
 manual_ham.append([sum(int(u!=v) for u,v in zip(q['bits'],r['bits'])) for r in a])
 manual_color.append([math.sqrt(max(0,1-math.fsum(math.sqrt(u*v) for u,v in zip(q['hist'],r['hist'])))) for r in a])
assert np.array_equal(manual_ham,arrays['hamming'])
color_gap=float(np.max(np.abs(np.array(manual_color)-arrays['hellinger'])))
assert color_gap<1e-12
for i,q in enumerate(comparison['per_image']):
 for method,dist in [('phash',manual_ham),('color',manual_color)]:
  top=sorted(range(340),key=lambda j:dist[i][j])[:3]
  assert [rows['aihub'][j]['name'] for j in top]==[r['name'] for r in q[method]]
  for j,r in zip(top,q[method]): assert abs(dist[i][j]-r['distance'])<1e-12
counter={k:dict(collections.Counter(str(m.get(k)) for m in metas)) for k in ['farm_id','kind_type','growth_stage']}
for k,v in counter.items(): assert v==profile['metadata'][k]
assert not(set(r['sha'] for r in a)&set(r['sha'] for r in b))
assert not(set(r['pixel'] for r in a)&set(r['pixel'] for r in b))
assert min(min(r) for r in manual_ham)==comparison['min_phash']==10
assert sum(x<=6 for r in manual_ham for x in r)==comparison['pairs_phash_le6']==0
# Internal nearest-neighbor Hamming via manual XOR of packed bit integers.
packed=[sum(bit<<k for k,bit in enumerate(r['bits'])) for r in a]
internal_nn=[min((x^y).bit_count() for j,y in enumerate(packed) if j!=i) for i,x in enumerate(packed)]
assert np.array_equal(internal_nn,arrays['aihub_internal_phash'].min(axis=1))
result={'status':'PASS','fresh_images':361,'fresh_jsons':340,'cross_pairs':7140,'byte_pixel_size_phash_color_all_recomputed':True,'manual_cosine_transform_all_pass':True,'manual_hamming_all_pass':True,'manual_hellinger_max_gap':color_gap,'farm_kind_stage':counter,'metadata_dimension_mismatches':len(mismatches),'dimension_mismatch_examples':mismatches[:8],'public_labels':{'real':3,'positive':18},'real_nn_phash':[min(manual_ham[i]) for i in range(3)],'internal_nn_phash_median':float(np.median(internal_nn)),'cross_exact_duplicates':0,'phash_le6_pairs':0,'caution':'pHash low distance not source proof; high distance does not exclude transformed source; no semantic embedding/model performance'}
with (HERE/'verification_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False),flush=True)
