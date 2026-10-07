"""Three frozen endpoint rules. No evaluator labels, future query, or CSV access."""
import math
from collections import defaultdict
from blk_context_v1 import RAW, key

ALPHA = 0.2
CHAIN_MAX_EC_GAP = 0.05
CHAIN_EC_MARGIN = 0.005
INPUT_COVERAGE = 0.75
INPUT_RELATIVE_MARGIN = 0.1

def finite(x):
    return x is not None and math.isfinite(x)

class EndpointMethods:
    def __init__(self, layout, reference_inputs, reference_labels):
        assert set(reference_inputs) == set(reference_labels) == set(layout['train_ids'])
        self.inputs = reference_inputs
        self.labels = reference_labels
        self.blocks = {}
        for b, a in zip(layout['blocks'], layout['endpoint_anchor_ids']):
            assert b['farm'] == key(a['left_23h'])[0] == key(a['right_0h'])[0]
            for day in b['query_days']:
                self.blocks[b['farm'], day] = (b, a)
        self.days = defaultdict(dict)
        for rid, obs in reference_inputs.items():
            f, d, h = key(rid)
            self.days[f, d][h] = obs
        self.scale = {}
        for c in RAW:
            xs = [o[c] for o in reference_inputs.values() if finite(o[c])]
            mu = math.fsum(xs) / len(xs)
            sd = math.sqrt(math.fsum((v-mu)**2 for v in xs)/len(xs))
            self.scale[c] = max(sd, 1e-8)
        self.parent = {d: d for d in self.days}
        self.links = []
        self._chains()

    def root(self, d):
        while self.parent[d] != d:
            d = self.parent[d]
        return d

    def _chains(self):
        # Mutual best endpoint match, train-only. Self links forbidden; ties abstain.
        records = sorted(self.days)
        best_out, best_in = {}, {}
        for source in records:
            f, d = source
            end = self.labels[f'{f}_{d:03d}_23']['sub_ec']
            options = sorted((abs(end-self.labels[f'{g}_{e:03d}_00']['sub_ec']), (g,e))
                             for g,e in records if g == f and e != d)
            if len(options) >= 2 and options[0][0] <= CHAIN_MAX_EC_GAP and options[1][0]-options[0][0] >= CHAIN_EC_MARGIN:
                best_out[source] = options[0][1]
        for target in records:
            f, d = target
            start = self.labels[f'{f}_{d:03d}_00']['sub_ec']
            options = sorted((abs(start-self.labels[f'{g}_{e:03d}_23']['sub_ec']), (g,e))
                             for g,e in records if g == f and e != d)
            if len(options) >= 2 and options[0][0] <= CHAIN_MAX_EC_GAP and options[1][0]-options[0][0] >= CHAIN_EC_MARGIN:
                best_in[target] = options[0][1]
        for source, target in sorted(best_out.items()):
            if best_in.get(target) == source:
                self.links.append((source, target))
                a, b = self.root(source), self.root(target)
                if a != b:
                    self.parent[max(a,b)] = min(a,b)

    def signature(self, observations, hour):
        out = []
        for c in RAW:
            vals = [observations[h][c] for h in range(hour+1)
                    if h in observations and finite(observations[h][c])]
            out.append(math.fsum(vals)/len(vals) if vals else None)
            out.append(observations.get(hour, {}).get(c))
        return out

    def assigned_chain(self, rid, prefix):
        f, d, h = key(rid)
        assert all(key(k)[0] == f and key(k)[1:] <= (d,h) for k in prefix)
        obs = {key(k)[2]: v for k,v in prefix.items() if key(k)[1] == d}
        q = self.signature(obs,h)
        best = {}
        for day, observations in self.days.items():
            if day[0] != f:
                continue
            x = self.signature(observations,h)
            diffs = [((a-b)/self.scale[RAW[i//2]])**2 for i,(a,b) in enumerate(zip(q,x)) if finite(a) and finite(b)]
            if len(diffs) < INPUT_COVERAGE*len(q):
                continue
            distance = math.sqrt(math.fsum(diffs)/len(diffs))
            root = self.root(day)
            best[root] = min(best.get(root,float('inf')),distance)
        ranked = sorted((v,k) for k,v in best.items())
        if len(ranked)<2:
            return None
        first, second = ranked[0][0],ranked[1][0]
        if second-first < INPUT_RELATIVE_MARGIN*max(second,1e-8):
            return None
        return ranked[0][1]

    def predict(self, rid, baseline, prefix):
        f,d,h = key(rid)
        b,a = self.blocks[f,d]
        left,right = a['left_23h'],a['right_0h']
        lf,ld,lh = key(left); rf,rd,rh = key(right)
        weight = ((24*d+h)-(24*ld+lh))/((24*rd+rh)-(24*ld+lh))
        assert 0 < weight < 1
        past = self.labels[left]['sub_ec']
        both = (1-weight)*past+weight*self.labels[right]['sub_ec']
        assigned = self.assigned_chain(rid,prefix)
        same = self.root((lf,ld)) == self.root((rf,rd)) == assigned
        blend = lambda level: (1-ALPHA)*baseline+ALPHA*level
        return {'PAST_ENDPOINT':blend(past),'BOTH_ENDPOINT':blend(both),
                'CHAIN_PREFIX_GUARD':blend(both) if same else baseline,
                'chain_guard_active':same,'calendar_weight':weight}
