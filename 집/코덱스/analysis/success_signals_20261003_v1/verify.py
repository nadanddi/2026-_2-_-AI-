from diagnose import *

def main():
    d=pd.read_csv(HERE/'event_days.csv');r=json.loads((HERE/'result.json').read_text(encoding='utf-8'))
    raw=pd.read_csv(ROOT/'공용/대회자료/정형데이터/참가자_배포/train_X.csv',usecols=['row_id']+RAW)
    raw=raw[raw.row_id.str[:3].isin(['F13','F47'])];groups={}
    for row in raw.itertuples():groups.setdefault((row.row_id[:3],int(row.row_id[4:7])),[]).append(row)
    def avg(values):
        values=[float(v) for v in values if pd.notna(v)]
        return math.fsum(values)/len(values) if values else float('nan')
    def std(values):
        values=[float(v) for v in values if pd.notna(v)]
        if not values:return float('nan')
        m=math.fsum(values)/len(values)
        return math.sqrt(math.fsum((v-m)**2 for v in values)/len(values))
    rawmeans={k:np.array([avg(getattr(a,col) for a in rows) for col in RAW]) for k,rows in groups.items()}
    checks=0;distancechecks=0;maxdiff=0.
    for row in d.itertuples():
        rows=groups[(row.farm,int(row.day))];night=[a for a in rows if int(a.row_id[8:10])<=6];day=[a for a in rows if 9<=int(a.row_id[8:10])<=16];h0=[a for a in rows if a.row_id.endswith('_00')];assert len(h0)==1
        expected={col+'_mean':avg(getattr(a,col) for a in rows) for col in RAW}
        for col in ['in_temp','in_hum','in_co2','out_temp']:expected[col+'_std']=std(getattr(a,col) for a in rows)
        for col in ['in_temp','in_hum']:expected[col+'_day_minus_night']=avg(getattr(a,col) for a in day)-avg(getattr(a,col) for a in night)
        expected['vent_zero_fraction']=sum(a.act_vent==0 for a in rows)/len(rows);expected['night_heating']=avg(a.act_heating for a in night)
        for col in ['act_heating','act_thermal','act_vent','in_temp','in_hum']:expected[col+'_h0']=float(getattr(h0[0],col))
        for col,val in expected.items():
            observed=float(getattr(row,col))
            if math.isnan(val):assert math.isnan(observed)
            else:maxdiff=max(maxdiff,abs(val-observed));assert abs(val-observed)<1e-10
            checks+=1
        prefix=f'{"T" if row.target=="TEMP" else "E"}_DIAG10_{int(row.fold)}'
        z=np.load(STUDY/f'{prefix}_cpu.npz');keys=sorted({(v[:3],int(v[4:7])) for v in z['outer_train_id']});assert (row.farm,int(row.day)) not in keys
        matrix=np.array([rawmeans[k] for k in keys]);median=np.nanmedian(matrix,axis=0);filled=np.where(np.isnan(matrix),median,matrix);mean=np.array([avg(filled[:,j]) for j in range(14)]);sd=np.array([std(filled[:,j]) for j in range(14)]);sd[sd==0]=1
        query=np.where(np.isnan(rawmeans[(row.farm,int(row.day))]),median,rawmeans[(row.farm,int(row.day))])
        candidates=[]
        for k,v in zip(keys,filled):
            if k[0]==row.farm and (k[1]>=179)==(row.day>=179):candidates.append((math.sqrt(math.fsum(float(a)**2 for a in (v-query)/sd)/14),k))
        candidates.sort();distance,key=candidates[0]
        assert abs(distance-row.nearest_train_distance)<1e-10
        saved=pd.read_csv(HERE/'neighbors.csv');saved=saved[(saved.target==row.target)&(saved.farm==row.farm)&(saved.day==row.day)&(saved['rank']==1)].iloc[0];assert key==(saved.neighbor_farm,int(saved.neighbor_day))
        distancechecks+=1
    pchecks=0;contrastchecks=0;matchedchecks=0
    for target,q in d.groupby('target'):
        q=q.reset_index(drop=True);labels=q.good.to_numpy(bool);strata={}
        for i,row in enumerate(q.itertuples()):strata.setdefault((row.farm,int(row.day>=179),int(np.sign(row.gap)) if target=='TEMP' else 0),[]).append(i)
        options=[list(itertools.combinations(ids,int(labels[ids].sum()))) for ids in strata.values()];masks=[]
        for choice in itertools.product(*options):
            mask=np.zeros(len(q));mask[[i for a in choice for i in a]]=1;masks.append(mask)
        masks=np.array(masks);assert len(masks)<=20000
        operator=masks/labels.sum()-(1-masks)/(~labels).sum()
        for comp in [a for a in r['comparisons'] if a['target']==target]:
            a=q[comp['feature']].to_numpy(float);pos=a[labels];neg=a[~labels]
            delta=avg(pos)-avg(neg);assert abs(delta-comp['difference'])<1e-10
            percentiles=[sum(float(v)<=float(u) for v in neg)/len(neg) for u in pos]
            assert max(abs(v-w) for v,w in zip(percentiles,comp['good_bad_percentiles']))<1e-12
            contrastchecks+=1
            if comp['p'] is None:continue
            contrasts=operator@a;observed=abs(delta);p=float(np.mean(np.abs(contrasts)>=observed-1e-12));assert abs(p-comp['p'])<1e-12;assert abs(min(1,60*p)-comp['p_bonferroni'])<1e-12;pchecks+=1
        for match in [a for a in r['matches'] if a['target']==target]:
            good=q[(q.farm==match['farm'])&(q.day==match['day'])].iloc[0];negative=q[~q.good];avail=negative[(negative.farm==good.farm)&((negative.day>=179)==(good.day>=179))]
            if target=='TEMP':avail=avail[np.sign(avail.gap)==np.sign(good.gap)];severity=avail.gap
            else:severity=avail.y
            order=sorted(zip(abs(severity-(good.gap if target=='TEMP' else good.y)),avail.day))[:3];assert [int(v) for _,v in order]==match['match_days']
            chosen=avail[avail.day.isin(match['match_days'])];col=match['feature']
            if len(chosen):assert abs(avg(chosen[col])-match['matched_bad_mean'])<1e-10
            else:assert match['matched_bad_mean'] is None
            matchedchecks+=1
    positives=d[d.good]
    e=pd.read_csv(ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv');e=e[(e.validator=='DIAG10')&(e.seed==7)]
    member_cases=[]
    for farm,day in [('F13',177),('F47',139)]:
        q=e[(e.farm==farm)&(e.day==day)];vals={col:avg(q[col]) for col in ['sub_ec','season_v2','season_r3','season_pfn']}
        assert abs(.8*vals['season_r3']+.2*vals['season_pfn']-vals['season_v2'])<1e-12
        member_cases.append(dict(farm=farm,day=day,**vals))
    result=dict(status='PASS',raw_input_cells=checks,nearest_train_checks=distancechecks,max_input_difference=maxdiff,contrast_checks=contrastchecks,stratified_exact_permutation_checks=pchecks,matched_comparison_checks=matchedchecks,significant_bonferroni=sum(a['p_bonferroni'] is not None and a['p_bonferroni']<.05 for a in r['comparisons']),member_cases=member_cases,methods=['raw inputs/math.fsum','manual outertrain imputation/variance/distance','exact stratified permutation contrast matrix','independent same-stratum severity matching','EC mixture mean identity'])
    (HERE/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
