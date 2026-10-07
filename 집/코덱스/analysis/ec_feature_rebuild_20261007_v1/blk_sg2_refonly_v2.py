"""Same SG2 policy; require complete prediction prefix and actual pivot column order."""
from blk_sg2_refonly_v1 import *

class RefOnlySG2(RefOnlySG2):
    def twin(self,f,d,h,obs):
        dates=sorted(e for g,e in self.ref if g==f)
        cols=np.asarray(self.S['hrs']<=h)
        A=self.S['WV'].loc[[(f,e) for e in dates]].values[:,cols]
        selected=self.S['WV'].columns[cols]
        b=np.array([(obs[i][c]-self.mu[f,c])/self.sd[f,c] if i in obs and obs[i][c] is not None else np.nan for c,i in selected])
        assert A.shape[1]==len(b)
        dist=np.sqrt(np.nanmean((A-b)**2,axis=1))
        match=dist<=.05
        return float(np.mean([self.cal[f,e] for e,m in zip(dates,match) if m])) if match.any() else None

    def predict_one(self,rid,baseline_prefix,scope):
        f,d,h=key(rid)
        assert set(baseline_prefix)=={f'{f}_{d:03d}_{j:02d}' for j in range(h+1)},'incomplete prediction prefix'
        assert all(np.isfinite(v) for v in baseline_prefix.values())
        return super().predict_one(rid,baseline_prefix,scope)
