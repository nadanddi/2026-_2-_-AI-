from support import *
from analyze import fitmeta,pfn
import math
lab,_,_,_,folds,outer=loadtemp();name,k,fd=folds[0];_,vm=common.split_mask(lab,fd);va=lab[vm];z=dict(np.load(OUT/'T_DIAG10_0_cpu.npz'));records=[]
for seed in TSEEDS:
    for context,seeds in [('1-8',list(range(1,9))),('17-24',list(range(17,25)))]:
        p=np.column_stack([z[f'base_{seed}'],z[f'codex_{seed}'],pfn('T_DIAG10_0',seeds,z['row_id'])]);d=outer[(outer.validator==name)&(outer.base_seed==seed)&(outer.context==context)];values=[d[d.member==member].set_index('row_id').prediction.reindex(va.row_id).to_numpy() for member in ['BASE','CODEX','PFN','W30G']];qp=np.column_stack(values[:3]);bias,gated,audit=fitmeta(z,p,qp,conditions(va),gate(va));assert np.isfinite(bias).all() and np.isfinite(gated).all();ref=values[3];cold=gate(va)==0;assert np.max(np.abs(bias[cold]-ref[cold]),initial=0)<1e-12;assert np.max(np.abs(gated[cold]-ref[cold]),initial=0)<1e-12
        y=va.sub_temp.to_numpy();rm=lambda pr:math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(pr,y))/len(y));r0=rm(ref)
        records.append(dict(seed=seed,context=context,fold='DIAG10/0',baseline_rmse=r0,bias_delta_pct=100*(rm(bias)/r0-1),gate_delta_pct=100*(rm(gated)/r0-1)))
savej(HERE/'smoke_temp.json',dict(status='PASS',scope='first fold only; not adoption evidence',records=records,source_hash=sha(__file__)));print(json.dumps(records),flush=True)
