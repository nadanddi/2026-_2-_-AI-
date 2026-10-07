"""Freeze input-only SG2 reference selections; apply unchanged prediction rule."""
from blk_baseline_data_v1 import np, key, sha, HERE
from blk_sg2_refonly_v2 import RefOnlySG2
from types import MappingProxyType
import math

SCOPES=('BLK_QUERY_ROLE','BLK_RAW_PASS')

class SG2InputPlan:
    def __init__(self,ctx,ids):
        self.sg=RefOnlySG2(ctx)
        self.ids=tuple(ids)
        choices={}
        for scope in SCOPES:
            for rid in self.ids:
                f,d,h=key(rid)
                # Selection is independent of prediction values in source SG2.
                zero={f'{f}_{d:03d}_{j:02d}':0. for j in range(h+1)}
                _,diag=self.sg.predict_one(rid,zero,scope)
                chosen=diag.get('reference_day')
                level=self.sg.ec[f,chosen] if chosen is not None else None
                assert level is None or math.isfinite(level)
                choices[scope,rid]=MappingProxyType({'active':diag['active'],
                    'has_candidate':diag['has_candidate'],'reference_day':chosen,'level':level})
        self.choices=MappingProxyType(choices)
        self.source_sha256=sha(HERE/'blk_sg2_refonly_v1.py')

    def apply(self,rid,prefix,scope):
        f,d,h=key(rid)
        assert set(prefix)=={f'{f}_{d:03d}_{j:02d}' for j in range(h+1)}
        assert all(math.isfinite(value) for value in prefix.values())
        choice=self.choices[scope,rid]
        value=float(prefix[rid])
        if not choice['has_candidate']:return value
        mean=float(np.mean(list(prefix.values())))
        difference=choice['level']-mean
        return value+.5*difference if abs(difference)<=.30 else value

    def audit_source(self,rid,prefix,scope):
        source,_=self.sg.predict_one(rid,prefix,scope)
        planned=self.apply(rid,prefix,scope)
        assert abs(source-planned)<1e-12
        return abs(source-planned)
