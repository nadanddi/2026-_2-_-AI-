import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,hashlib,math,collections
import numpy as np
from PIL import Image,ImageOps
import torch,torchvision
RUN=Path(__file__).resolve().parent;COMP=ROOT/'공용/대회자료/이미지데이터/참가자_배포/공개샘플/images'
selection=json.loads((RUN/'selection_v1.json').read_text(encoding='utf-8'));prepared=json.loads((RUN/'prepared_selection_v1.json').read_text(encoding='utf-8'))
rows=selection['candidates'];assert len(rows)==1000 and len({r['source_member'] for r in rows})==1000
assert sum(not r['excluded_reasons'] for r in rows)==selection['eligible']==513
tainted=[r for r in rows if any(x in r['excluded_reasons'] for x in ['near_public_phash_candidate','public_exact_byte','prior_public_scene_family_suspect'])]
prefixes={r['prefix_candidate'] for r in tainted};components={r['component'] for r in tainted}
def pixelsha(im):return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
public_pixels={}
for p in COMP.glob('*.jpg'):
    with Image.open(p) as im:im=ImageOps.exif_transpose(im).convert('RGB');im.load();public_pixels[pixelsha(im)]=p.name
torch.set_num_threads(2);model=torchvision.models.resnet18(weights=None)
asset=json.loads((RUN.parent/'image_ingestion_20261010_v1/pretrained_asset_v1.json').read_text(encoding='utf-8'));model.load_state_dict(torch.load(asset['path'],weights_only=True,map_location='cpu'));model.fc=torch.nn.Identity();model.eval()
def feature(im):
    a=np.array(im.resize((224,224),Image.Resampling.BILINEAR),copy=True);x=torch.tensor(a).permute(2,0,1).float()/255
    x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    with torch.inference_mode():v=model(x[None])[0].numpy()
    assert np.isfinite(v).all() and np.linalg.norm(v)>0
    return v/np.linalg.norm(v)
def hist_manual(im):
    pixels=list(im.resize((128,128),Image.Resampling.BILINEAR).getdata());counts=[0]*24
    for rgb in pixels:
        for c,value in enumerate(rgb):counts[c*8+value//32]+=1
    return [v/(3*len(pixels)) for v in counts]
checks=[]
for r in prepared['sources']:
    assert r['prefix_candidate'] not in prefixes and r['component'] not in components
    assert r['magenta']<=.35 and r['dimension_match'] and not r['excluded_reasons']
    raw=Path(r['original_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['payload_sha256']
    with Image.open(r['original_path']) as im:im=ImageOps.exif_transpose(im).convert('RGB');im.load()
    assert pixelsha(im) not in public_pixels
    with Image.open(COMP/r['selection_reference']) as ref:ref=ImageOps.exif_transpose(ref).convert('RGB');ref.load()
    a,b=hist_manual(im),hist_manual(ref)
    color=math.sqrt(max(0,1-sum(math.sqrt(x*y) for x,y in zip(a,b))))
    cosine=1-float(np.dot(feature(im),feature(ref)))
    assert abs(color-r['color_Hellinger'])<1e-12 and abs(cosine-r['embedding_cos_distance'])<1e-5
    score=.5*cosine+.3*color+.2*r['phash_distance']/63
    assert abs(score-r['score'])<1e-5
    checks.append({'image_id':r['image_id'],'exact_decoded_pixel_duplicate_public':False,'color_independent':color,'embedding_fresh_CPU':cosine,'ranking_score_fresh':score,'component_taint':False})
for ranking in selection['rankings']:
    assert all(math.isfinite(r[k]) for r in ranking['top5'] for k in ['score','embedding_cos_distance','color_Hellinger','phash_distance'])
result={'all_checks_pass':True,'source_checks':checks,'eligible_candidates':513,'searched_candidates':1000,'public_images':21,'tainted_prefixes':sorted(prefixes),'tainted_components':sorted(components),'exact_pixel_check_scope':'selected2 original fullresolution versus public21 only; not all1000','numerical_checks':'selected scores freshCPU embedding/manual RGB histogram/Bhattacharyya identity; all recordedtop5 scoresfinite','public_observation_used':True,'scope':'selection QA only, no performance/independence claim'}
with (RUN/'selection_verification_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
