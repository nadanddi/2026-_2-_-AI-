import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
import json,time,hashlib,shutil,subprocess
import torch,torchvision,numpy,PIL
from torchvision.models import resnet18
RUN=Path(__file__).resolve().parent
out=RUN/'runtime_smoke_v2.json'
if out.exists():raise RuntimeError('immutable output')
torch.manual_seed(20261010)
result={'scope':'environment and random-weight synthetic tensor smoke only; no real-image model trained, no accuracy measured','python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':numpy.__version__,'PIL':PIL.__version__,'module_paths':{'torch':torch.__file__,'torchvision':torchvision.__file__,'numpy':numpy.__file__,'PIL':PIL.__file__},'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,'seed':20261010,'dummy_training_batches':[]}
assert torch.cuda.is_available(),'CUDA unavailable'
result['gpu']=torch.cuda.get_device_name(0);result['device_total_bytes']=torch.cuda.get_device_properties(0).total_memory
model=resnet18(weights=None);model.fc=torch.nn.Linear(512,1);model.cuda()
for batch in [2,4]:
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);scaler=torch.amp.GradScaler('cuda');x=torch.randn(batch,3,384,384,device='cuda');y=torch.zeros(batch,1,device='cuda');start=time.perf_counter()
    try:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            pred=model(x);loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,y)
        assert bool(torch.isfinite(loss))
        parameter_before=model.fc.weight.detach().clone();scale_before=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer);gradients_finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None);assert gradients_finite;scaler.step(optimizer);scaler.update();torch.cuda.synchronize();parameter_changed=not torch.equal(parameter_before,model.fc.weight.detach());optimizer_step=int(optimizer.state[model.fc.weight]['step'].item());assert parameter_changed and optimizer_step==1
        result['dummy_training_batches'].append({'batch':batch,'size':384,'AMP':True,'finite_loss':True,'gradients_finite':gradients_finite,'fc_weight_changed':parameter_changed,'optimizer_step':optimizer_step,'scale_before':scale_before,'scale_after':scaler.get_scale(),'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'GPU_free_bytes_after_step':torch.cuda.mem_get_info()[0]})
    except torch.cuda.OutOfMemoryError:
        result['dummy_training_batches'].append({'batch':batch,'oom':True})
    del x,y,optimizer,scaler
    model.zero_grad(set_to_none=True)
model.cpu().eval();torch.cuda.empty_cache()
with torch.inference_mode():p=model(torch.zeros(1,3,384,384)).sigmoid()
result['cpu_fallback_finite']=bool(torch.isfinite(p).all());assert result['cpu_fallback_finite']
check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True);result['pip_check']={'scope':'venv package metadata','returncode':check.returncode,'stdout':check.stdout.strip(),'stderr':check.stderr.strip()};assert check.returncode==0
result['free_disk_bytes_after']=shutil.disk_usage(ROOT).free
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False),flush=True)

