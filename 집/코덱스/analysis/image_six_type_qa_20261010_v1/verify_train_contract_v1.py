import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,hashlib,collections,shutil
import numpy as np
from PIL import Image
import torch,torchvision
RUN=Path(__file__).resolve().parent
def read(name):return json.loads((RUN/name).read_text(encoding='utf-8'))
def sha(path):return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()
def state_sha(state):
    h=hashlib.sha256()
    for k,t in state.items():
        h.update(k.encode());h.update(str(t.dtype).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
torch.set_num_threads(2)
registration=read('training_registration_v1.json');resume=read('resume_registration_v1.json')
for name,digest in resume['pinned_files'].items():assert sha(RUN/name)==digest
assert registration['qa_verification_sha256']==sha(RUN/'qa_verification_v2.json')
assert read('qa_verification_v2.json')['all_checks_pass']
preflight=read('resume_preflight_v1.json')
assert preflight['validation_PASS'] and preflight['registered_validation_sha_match']
manifest=read('qa_manifest_v1.json');rows=manifest['rows'];byid={r['sample_id']:r for r in rows}
assert len(rows)==len(byid)==14
bad=set(json.loads((RUN.parent/'image_source_mapping_20261010_v1/quarantine_v2.json').read_text(encoding='utf-8'))['image_ids'])
xs=[]
for row in rows:
    assert sha(row['path'])==row['sha256'] and row['source_id'] not in bad and not row['withheld']
    with Image.open(row['path']) as im:arr=np.array(im.convert('RGB').resize((384,384),Image.Resampling.BILINEAR),copy=True)
    raw=torch.tensor(arr).permute(2,0,1).float().div(255)
    xs.append((raw-torch.tensor([.485,.456,.406]).view(3,1,1))/torch.tensor([.229,.224,.225]).view(3,1,1))
x=torch.stack(xs)
first=read('train_first_v1.json');final=read('train_finish_v1.json')
states=[]
for receipt,steps in [(first,1),(final,12)]:
    assert sha(receipt['checkpoint'])==receipt['checkpoint_sha256']
    st=torch.load(receipt['checkpoint'],map_location='cpu',weights_only=True)
    assert st['completed_steps']==receipt['completed_steps']==steps
    assert st['seed']==20261010 and st['manifest_sha256']==sha(RUN/'qa_manifest_v1.json')
    assert len(st['history'])==steps and st['history']==receipt['history']
    assert all(torch.isfinite(t).all() for t in st['model'].values())
    assert state_sha(st['model'])==receipt['state_sha256']
    assert {int(v['step']) for v in st['optimizer']['state'].values()}=={steps}
    states.append(st)
assert final['history'][:1]==first['history'] and first['state_sha256']!=final['state_sha256']
counts=collections.Counter();types=collections.Counter()
for number,event in enumerate(final['history'],1):
    assert event['step']==number and event['labels']==[0,0,1,1]
    assert event['optimizer_steps']==[number] and event['fc_weight_changed']
    assert event['all_gradients_finite'] and event['post_clip_gradients_finite']
    for sid,label in zip(event['sample_ids'],event['labels']):
        row=byid[sid];assert row['label']==label;counts[sid]+=1;types[row['positive_type']]+=1
assert sorted(counts.values())==[2]*12+[12]*2 and types['real']==24
assert all(v==4 for k,v in types.items() if k!='real')
model=torchvision.models.resnet18(weights=None);model.fc=torch.nn.Linear(512,1)
model.load_state_dict(states[1]['model']);model.eval();before=state_sha(model.state_dict())
with torch.inference_mode():
    p=torch.cat([model(x[i:i+2]).sigmoid().flatten() for i in range(0,14,2)]).numpy()
    reverse=torch.cat([model(x.flip(0)[i:i+5]).sigmoid().flatten() for i in range(0,14,5)]).numpy()[::-1]
contract=read('cpu_inference_contract_v1.json')
differences={'fresh_batch2_vs_saved':float(np.max(np.abs(p-np.array(contract['outputs'])))),
             'fresh_reverse_batch5':float(np.max(np.abs(p-reverse)))}
assert all(d<=1e-5 for d in differences.values())
assert before==state_sha(model.state_dict())==final['state_sha256']
assert p.shape==(14,) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
dirs=[ROOT/'집/코덱스/local'/n for n in ['image_ingestion_20261010_v1','image_runtime_20261010_v1','image_download_complete_20261010_v1','image_source_mapping_20261010_v1','image_six_type_qa_20261010_v1']]
sizes={d.name:sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) for d in dirs}
assert sum(sizes.values())<16*1024**3 and sizes[dirs[-1].name]<1024**3
free=shutil.disk_usage(ROOT).free;assert free>30*1024**3
out={'all_checks_pass':True,'verified_steps':12,'unique_images':14,'real_source_photos':2,
     'sampling_draws':sum(counts.values()),'draws_by_type':dict(types),'fresh_CPU_differences':differences,
     'CPU_state_unchanged':True,'checkpoint_bytes':[Path(r['checkpoint']).stat().st_size for r in [first,final]],
     'cache_bytes_by_task':sizes,'total_work_cache_bytes':sum(sizes.values()),'C_free_bytes':free,
     'QA_gate_process':'first used QA1; stronger registered QA2 enforced before resume; original records preserved',
     'leakage_scope':'train-only on development farms AIF005/AIF007; no holdout evaluation or claimed performance',
     'reproducibility_scope':'fixed inputs/code/checkpoint hashes and CPU replay verified; independent training rerun not performed',
     'limitations':['two sources only','individual original edit history unverified','overfit/provenance shortcuts unmeasured','no CV/accuracy/AUROC/adoption/submission']}
with (RUN/'train_verification_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
