"""Same frozen hypotheses; immutable reference copy, structural checks, final clip."""
from types import MappingProxyType
import math
from blk_endpoint_methods_v1 import EndpointMethods
from blk_context_v1 import key

class EndpointMethodsV2(EndpointMethods):
    def __init__(self,layout,inputs,labels):
        assert len(layout['blocks'])==len(layout['endpoint_anchor_ids'])
        for b,a in zip(layout['blocks'],layout['endpoint_anchor_ids']):
            assert b['query_days']==a['query_days']
            assert a['left_23h']==f'{b["farm"]}_{b["flank_left"][-1]:03d}_23'
            assert a['right_0h']==f'{b["farm"]}_{b["flank_right"][0]:03d}_00'
            assert a['left_23h'] in labels and a['right_0h'] in labels
        immutable=lambda x:MappingProxyType({k:MappingProxyType(dict(v)) for k,v in x.items()})
        super().__init__(layout,immutable(inputs),immutable(labels))
        assert all(set(obs)==set(range(24)) for obs in self.days.values())
        assert all(math.isfinite(v['sub_ec']) for v in self.labels.values())
        self.lo=min(v['sub_ec'] for v in self.labels.values())
        self.hi=max(v['sub_ec'] for v in self.labels.values())

    def predict(self,rid,baseline,prefix):
        assert math.isfinite(baseline)
        out=super().predict(rid,baseline,prefix)
        for c in ['PAST_ENDPOINT','BOTH_ENDPOINT','CHAIN_PREFIX_GUARD']:
            out[c]=max(self.lo,min(self.hi,out[c]))
        return out
