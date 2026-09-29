"""Extra causal features: within-day 'fingerprint' values at fixed earlier hours of the same day (NaN until that hour has passed)."""
from common import *
FPV=["out_temp","out_hum","out_wspd","in_temp","in_co2","in_hum"]
FPH=(0,3,6,9,12,15,18)
def add_fingerprint(F):
    F=F.copy()
    for v in FPV:
        piv=F.pivot_table(index=["gh","day"],columns="hr",values=v)
        for h in FPH:
            if h not in piv.columns: continue
            s=piv[h]
            val=s.reindex(pd.MultiIndex.from_arrays([F.gh,F.day])).values
            F[f"fp_{v}_{h}"]=np.where(F.hr.values>=h,val,np.nan)
    return F
