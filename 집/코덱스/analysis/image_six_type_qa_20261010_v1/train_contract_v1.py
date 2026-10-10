import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,hashlib,io,shutil,argparse,time
import numpy as np
from PIL import Image
import torch,torchvision
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local/image_six_type_qa_20261010_v1'
SEED=20261010
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_digest(model):
    h=hashlib.sha256()
    for key,t in model.state_dict().items():h.update(key.encode());h.update(str(t.dtype).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def save(name,obj):
    with (RUN/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def model_new():
    model=torchvision.models.resnet18(weights=None);model.fc=torch.nn.Linear(model.fc.in_features,1);return model
def data():
    frozen=json.loads((RUN/'training_registration_v1.json').read_text(encoding='utf-8'))
    assert digest(RUN/'qa_manifest_v1.json')==frozen['manifest_sha256']
    o=json.loads((RUN/'qa_manifest_v1.json').read_text(encoding='utf-8'));rows=o['rows'];xs=[]
    verified=json.loads((RUN/'qa_verification_v1.json').read_text(encoding='utf-8'));assert verified['all_checks_pass']
    bad=set(json.loads((RUN.parent/'image_source_mapping_20261010_v1/quarantine_v2.json').read_text(encoding='utf-8'))['image_ids'])
    assert len(rows)==14 and all(r['source_id'] not in bad and r['farm_id'] in ['AIF005','AIF007'] for r in rows)
    for r in rows:
        assert digest(r['path'])==r['sha256'] and not r['withheld']
        with Image.open(r['path']) as im:arr=np.asarray(im.convert('RGB').resize((384,384),Image.Resampling.BILINEAR)).copy()
        x=torch.from_numpy(arr).permute(2,0,1).float()/255
        xs.append((x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None])
    return rows,torch.stack(xs),torch.tensor([r['label'] for r in rows],dtype=torch.float32)
def checkpoint(name,obj):
    maxbytes=200*1024**2
    dirs=[ROOT/'집/코덱스/local'/n for n in ['image_ingestion_20261010_v1','image_runtime_20261010_v1','image_download_complete_20261010_v1','image_source_mapping_20261010_v1','image_six_type_qa_20261010_v1']]
    size=lambda p:sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    if size(LOCAL)+maxbytes>1024**3 or sum(size(p) for p in dirs)+maxbytes>16*1024**3 or shutil.disk_usage(ROOT).free-maxbytes<30*1024**3:raise ValueError('checkpoint budget')
    p=LOCAL/name
    with p.open('xb') as f:torch.save(obj,f)
    if p.stat().st_size>maxbytes:raise ValueError('checkpoint unexpectedly exceeded reserved bytes; preserve and stop')
    return str(p)
def fit(stage):
    torch.set_num_threads(2);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('registered GPU training required')
    rows,x,y=data();device=torch.device('cuda');real=[i for i,r in enumerate(rows) if r['label']==0];positive=[[i for i,r in enumerate(rows) if r['label']==1 and r['source_id']==sid] for sid in ['415785','833421']]
    m=model_new().to(device);optimizer=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda');history=[];start=0
    if stage=='first':
        asset=json.loads((RUN.parent/'image_ingestion_20261010_v1/pretrained_asset_v1.json').read_text(encoding='utf-8'));assert digest(asset['path'])==asset['sha256']
        initial=torch.load(asset['path'],map_location='cpu',weights_only=True);initial.pop('fc.weight');initial.pop('fc.bias');missing=m.load_state_dict(initial,strict=False);assert missing.missing_keys==['fc.weight','fc.bias'] and missing.unexpected_keys==[]
    else:
        first=json.loads((RUN/'train_first_v1.json').read_text(encoding='utf-8'));assert digest(first['checkpoint'])==first['checkpoint_sha256']
        st=torch.load(first['checkpoint'],map_location='cpu',weights_only=True);m.load_state_dict(st['model']);optimizer.load_state_dict(st['optimizer']);scaler.load_state_dict(st['scaler']);history=st['history'];start=st['completed_steps'];torch.set_rng_state(st['rng_cpu']);torch.cuda.set_rng_state_all(st['rng_cuda'])
    m.train();began=time.perf_counter();torch.cuda.reset_peak_memory_stats();last=1 if stage=='first' else 12
    for step in range(start,last):
        ids=real+[positive[0][step%6],positive[1][step%6]];batch=x[ids].to(device);target=y[ids].to(device);optimizer.zero_grad(set_to_none=True);before=m.fc.weight.detach().clone();scale0=scaler.get_scale()
        with torch.autocast('cuda',dtype=torch.float16):logits=m(batch).flatten();loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
        if not torch.isfinite(loss):raise ValueError('loss nonfinite')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        grads=[p.grad for p in m.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads):raise ValueError('gradient nonfinite')
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);scaler.step(optimizer);scaler.update();changed=not torch.equal(before,m.fc.weight.detach())
        steps=sorted({int(v['step'].item()) for v in optimizer.state.values() if 'step' in v})
        if not changed or scaler.get_scale()<scale0 or steps!=[step+1]:raise ValueError('optimizer update/scale/step failed')
        item={'step':step+1,'sample_ids':[rows[i]['sample_id'] for i in ids],'labels':[rows[i]['label'] for i in ids],'loss':float(loss.detach().cpu()),'gradient_norm_before_clip':float(norm),'all_gradients_finite':True,'fc_weight_changed':changed,'optimizer_steps':steps,'scale_before':scale0,'scale_after':scaler.get_scale()};history.append(item);print(json.dumps(item),flush=True)
    cpu_state={k:t.detach().cpu().clone() for k,t in m.state_dict().items()};cpu_hash=state_digest(m)
    cp=checkpoint(f'train_contract_{stage}_v1.pt',{'model':cpu_state,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'completed_steps':last,'history':history,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'seed':SEED,'manifest_sha256':digest(RUN/'qa_manifest_v1.json')})
    receipt={'stage':stage,'completed_steps':last,'checkpoint':cp,'checkpoint_sha256':digest(cp),'state_sha256':cpu_hash,'history':history,'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.perf_counter()-began,'unique_input_files':14,'source_photos':2,'training_sampling':'2real+2positive perstep; six types cyclic/source paired','scope':'train-only pipeline contract, not validation or competition score','real_target_status':'camera-source provenance assumption; individual edit history unverified'}
    save('train_'+stage+'_v1.json',receipt)
    if stage=='finish':cpu_contract(m.cpu().eval(),x,cp)
def predict(model,x,batch):
    if len(x)==0:return np.empty((0,),dtype=np.float32)
    with torch.inference_mode():return torch.cat([torch.sigmoid(model(x[i:i+batch]).flatten()) for i in range(0,len(x),batch)]).numpy()
def cpu_contract(original,x,cp):
    before=state_digest(original);base=predict(original,x,3);reloaded=model_new();st=torch.load(cp,map_location='cpu',weights_only=True);reloaded.load_state_dict(st['model']);reloaded.eval();loaded_before=state_digest(reloaded)
    one=predict(reloaded,x,1);seven=predict(reloaded,x,7);rev=predict(reloaded,x.flip(0),3)[::-1].copy();empty=predict(reloaded,x[:0],1)
    differences={'reload_batch1':float(np.max(np.abs(base-one))),'batch7':float(np.max(np.abs(base-seven))),'reordered':float(np.max(np.abs(base-rev)))}
    assert all(v<=1e-5 for v in differences.values()) and before==state_digest(original)==loaded_before==state_digest(reloaded)
    assert base.shape==(14,) and len(empty)==0 and np.isfinite(base).all() and ((base>=0)&(base<=1)).all()
    save('cpu_inference_contract_v1.json',{'differences_CPU_to_CPU':differences,'tolerance':1e-5,'state_before_sha256':before,'parameter_and_buffer_unchanged':True,'eval_mode':True,'finite_probability_range':True,'empty_output_shape':list(empty.shape),'batch_sizes':[1,3,7],'outputs':base.tolist(),'scope':'in-sample contract outputs; no heldout predictions/accuracy/AUROC'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['first','finish']);fit(p.parse_args().stage)
