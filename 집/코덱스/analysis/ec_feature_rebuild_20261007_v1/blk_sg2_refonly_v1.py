"""SG2 ref-only fit and current/past query transform, explicit application scope."""
from blk_baseline_data_v1 import *
from blk_context_v1 import key

SG2_SOURCE=ROOT/'집/클로드/submission14_ec_sg2/sg2post.py'
sgns=dict(np=np,pd=pd,W=W)
extract(SG2_SOURCE,['_ident','prepare','ref_calendar'],sgns)

def signature(obs,h):
    def mean(c,start,end):
        a=[obs[i][c] for i in range(start,min(end,h)+1) if i in obs and obs[i][c] is not None]
        return np.mean(a) if a else np.nan
    def count(c):return sum((obs[i][c] or 0)>0 for i in range(h+1) if i in obs)
    vals=[o['in_temp'] for i,o in obs.items() if i<=h and o['in_temp'] is not None]
    opened=[i for i in range(h+1) if i in obs and (obs[i]['act_vent'] or 0)>0]
    # Match the source pandas Series order exactly.
    return dict(n_t=mean('in_temp',0,5),n_h=mean('in_hum',0,5),n_c=mean('in_co2',0,5),
                d_t=mean('in_temp',10,15),d_c=mean('in_co2',10,15),mx_t=max(vals) if vals else np.nan,
                th=count('act_thermal'),he=count('act_heating'),co=count('act_co2'),sh=count('act_shade'),ve=len(opened),
                fo=min(opened) if opened else 24,cf=mean('act_circfan',0,5))

class RefOnlySG2:
    def __init__(self,ctx):
        self.ctx=ctx
        self.x=dataframe(ctx.reference_inputs)
        self.S=sgns['prepare'](self.x)
        self.ref=set(self.S['WV'].index)
        self.cal=sgns['ref_calendar'](self.S,self.ref)
        self.mu,self.sd={},{}
        ident=sgns['_ident'](self.x)
        raw=ident.pivot_table(index=['farm','day'],columns='hour',values=W)
        for f in ['F13','F47']:
            p1=(raw.index.get_level_values(0)==f)&(raw.index.get_level_values(1)<179)
            for c in W:
                self.mu[f,c]=np.nanmean(raw.loc[p1,c].values)
                self.sd[f,c]=np.nanstd(raw.loc[p1,c].values)
                assert self.sd[f,c]>0
        self.ec={}
        for f,d in self.ref:
            self.ec[f,d]=np.mean([ctx.reference_labels[f'{f}_{d:03d}_{h:02d}']['sub_ec'] for h in range(24)])
        self.fit_signature={}
        for f in ['F13','F47']:
            for h in range(24):
                table=self.S['SIG'][h]
                rs=table.loc[[(g,d) for g,d in self.ref if g==f]].astype(float)
                self.fit_signature[f,h]=(rs.mean().values,rs.std().replace(0,np.nan).values)
        self.source_sha256=sha(SG2_SOURCE)

    def twin(self,f,d,h,obs):
        dates=sorted(e for g,e in self.ref if g==f)
        cols=np.asarray(self.S['hrs']<=h)
        A=self.S['WV'].loc[[(f,e) for e in dates]].values[:,cols]
        # pivot order is variables then hour, as in the original SG2.
        b=np.array([(obs[i][c]-self.mu[f,c])/self.sd[f,c] if i in obs and obs[i][c] is not None else np.nan
                    for c in sorted(W) for i in range(h+1)])
        dist=np.sqrt(np.nanmean((A-b)**2,axis=1))
        match=dist<=.05
        return float(np.mean([self.cal[f,e] for e,m in zip(dates,match) if m])) if match.any() else None

    def predict_one(self,rid,baseline_prefix,scope):
        assert scope in ['BLK_QUERY_ROLE','BLK_RAW_PASS']
        f,d,h=key(rid)
        assert rid in baseline_prefix
        assert all(key(k)[0]==f and key(k)[1]==d and key(k)[2]<=h for k in baseline_prefix)
        p=float(baseline_prefix[rid])
        if scope=='BLK_RAW_PASS' and d<179:return p,{'active':False,'changed':False,'has_candidate':False}
        prefix=self.ctx.query_prefix(rid)
        query={}
        for k,o in prefix.items():
            _,e,j=key(k);query.setdefault(e,{})[j]=o
        known=sorted(set(e for g,e in self.ref if g==f)|set(query))
        cache={}
        def full_date(e):
            if (f,e) in self.ref:return self.cal[f,e]
            assert e<d,'a full query day may only be earlier than current query'
            if e in cache:return cache[e]
            t=self.twin(f,e,23,query[e])
            if t is None:
                earlier=[a for a in known if a<e]
                second=False
                if e-1 in known:
                    # Source greedy pair status, constructed only through past full records.
                    def weather(a):
                        if a in query:return query[a]
                        return {i:self.ctx.reference_inputs[f'{f}_{a:03d}_{i:02d}'] for i in range(24)}
                    def is_second(a):
                        if a-1 not in known:return False
                        if is_second(a-1):return False
                        u,v=weather(a-1),weather(a)
                        pairs=[(u[i][c],v[i][c]) for c in W for i in range(24) if u[i][c] is not None and v[i][c] is not None]
                        return len(pairs)>=80 and max(abs(x-y) for x,y in pairs)<1e-9
                    second=is_second(e)
                t=full_date(earlier[-1])+(0 if second else .1) if earlier else 0.0
            cache[e]=t
            return t
        cq=self.twin(f,d,h,query[d]) if h>=5 else None
        if cq is None:
            prior=[e for e in known if e<d]
            cq=full_date(prior[-1])+.1 if prior else 0.0
        dates=sorted(e for g,e in self.ref if g==f and abs(self.cal[f,e]-cq)<=3 and self.cal[f,e]!=cq)
        if not dates:return p,{'active':True,'changed':False,'has_candidate':False}
        table=self.S['SIG'][h]
        mu,sd=self.fit_signature[f,h]
        sig=signature(query[d],h)
        rawq=np.array([sig[c] for c in table.columns],float)
        q=(rawq-mu)/sd
        use=np.isfinite(q)
        if not use.any():return p,{'active':True,'changed':False,'has_candidate':False}
        C=np.nan_to_num(((table.loc[[(f,e) for e in dates]].values.astype(float)-mu)/sd)[:,use])
        dist=np.sqrt(((C-q[use])**2).mean(axis=1))+.15*np.abs(np.array([self.cal[f,e] for e in dates])-cq)
        if not np.isfinite(dist).any():return p,{'active':True,'changed':False,'has_candidate':False}
        chosen=dates[int(np.nanargmin(dist))]
        a1=float(self.ec[f,chosen]);pm=float(np.mean(list(baseline_prefix.values())))
        corrected=p+.5*(a1-pm) if abs(a1-pm)<=.30 else p
        return corrected,{'active':True,'changed':corrected!=p,'has_candidate':True,'reference_day':chosen,'query_calendar':cq}
