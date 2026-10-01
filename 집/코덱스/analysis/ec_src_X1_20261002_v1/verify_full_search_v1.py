"""Independent arithmetic search: explicit pair/shift loops, math.fsum OLS.
Verification only; original precommitted search/thresholds unchanged.
"""
from pathlib import Path
import sys,csv,json,math,time
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
HERE=Path(__file__).resolve().parent
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
def mean(xs):return math.fsum(xs)/len(xs)
def main():
    start=time.monotonic()
    s=json.loads((HERE/'summary.json').read_text(encoding='utf-8'))
    locks={(v['farm'],int(v['day'])) for v in json.loads((Path(env.CODEX)/'ec_final_lock/locked_days.json').read_text(encoding='utf-8'))['selected']}
    data=defaultdict(dict)
    for r in read(Path(env.DATA)/'train_y.csv'):
        f,d,h=r['row_id'].split('_');d=int(d);h=int(h)
        if f not in ['F13','F47'] or (f,d) in locks:continue
        data[(f,d)][h]=float(r['sub_ec'])
    ys={k:[v[h] for h in range(24)] for k,v in data.items()}
    groups=defaultdict(list)
    good={}
    for r in read(HERE/'days.csv'):
        k=(r['farm'],int(r['day']));groups[r['stratum']].append(k)
        y=ys[k];m=mean(y);dif=[b-a for a,b in zip(y,y[1:])]
        good[k]=(max(y)-min(y)>=.03 and math.sqrt(mean([(z-m)**2 for z in y]))>=.008
                 and sum(abs(z)>=.000999 for z in dif)>=8 and len(set(round(z/.001) for z in dif))>=6)
        assert good[k]==(r['informative']=='True')
    counts=Counter();minima={};eligible=Counter()
    for name,keys in groups.items():
        for i,ki in enumerate(keys):
            if not good[ki]:continue
            for kj in keys[i+1:]:
                if not good[kj] or abs(ki[1]-kj[1])<7 or abs(mean(ys[ki])-mean(ys[kj]))>s['level_difference_limit']:continue
                eligible[name]+=1
                for shift in range(-6,7):
                    hs=[h for h in range(24) if 0<=h+shift<24]
                    y=[ys[ki][h] for h in hs];x=[ys[kj][h+shift] for h in hs]
                    mx=mean(x);my=mean(y)
                    vx=math.fsum((z-mx)**2 for z in x)
                    aa=math.fsum((u-mx)*(v-my) for u,v in zip(x,y))/vx if vx>0 else 0.
                    scale=math.fsum(u*v for u,v in zip(x,y))/math.fsum(u*u for u in x)
                    transforms=[('identity',1.,0.),('offset',1.,my-mx),('scale',scale,0.),('affine',aa,my-aa*mx)]
                    for family,a,b in transforms:
                        if not .5<=a<=2:continue
                        e=[v-(a*u+b) for u,v in zip(x,y)]
                        rmse=math.sqrt(mean([z*z for z in e]));ma=max(abs(z) for z in e)
                        key=f'{name}/{family}'
                        if key not in minima or rmse<minima[key]['rmse']:
                            minima[key]={'rmse':rmse,'maxabs':ma,'farm_i':ki[0],'day_i':ki[1],'farm_j':kj[0],'day_j':kj[1],'shift':shift,'a':a,'b':b}
                        if rmse<=.0005 and ma<=.001001:counts[key]+=1
    assert not counts and s['all_primary_pair_count']==0
    plateau=0;signatures=defaultdict(list)
    for k,y in ys.items():
        z=[round(v/.001) for v in y];h0=0
        for h in range(1,25):
            if h==24 or z[h]!=z[h0]:
                plateau+=int(h-h0>=3);h0=h
        for h in range(13):
            w=z[h:h+12];sig=tuple(b-a for a,b in zip(w,w[1:]));signatures[sig].append((k,h,max(w)-min(w)))
    repeated=[(sig,locs) for sig,locs in signatures.items() if len({loc[0] for loc in locs})>=2]
    assert plateau==s['plateau_runs_ge3h']==214 and len(repeated)==s['repeated_12h_diff_signatures']==1
    sample=[]
    for sig,locs in repeated:
        for k,h,span in locs:
            for hour in range(24):
                sample.append({'farm':k[0],'day':k[1],'hour':hour,'ec':ys[k][hour],
                               'in_repeated_window':h<=hour<h+12,'window_start':h,
                               'day_mean':mean(ys[k]),'day_range':max(ys[k])-min(ys[k]),'day_informative':good[k]})
    with (HERE/'repeated_window_raw_verified.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(sample[0]));w.writeheader();w.writerows(sample)
    near=[]
    for key,v in minima.items():near.append({'stratum_family':key,**v})
    with (HERE/'closest_primary_pairs_verified.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(near[0]));w.writeheader();w.writerows(near)
    result={'status':'PASS','method':'explicit nested pair/shift loops + math.fsum, separate from production matrix algebra',
            'primary_eligible_pairs_by_stratum':dict(eligible),'primary_matches_directly_researched':dict(counts),
            'informative_days_independent':sum(good.values()),'plateau_run_count_independent':plateau,
            'repeated_12h_signature_count_independent':len(repeated),
            'closest_rmse_per_stratum_family':minima,'seconds':time.monotonic()-start,
            'repeated_window_days':[{'farm':k[0],'day':k[1],'start':h,'span':span/1000,'mean':mean(ys[k]),
                                    'day_range':max(ys[k])-min(ys[k]),'day_informative':good[k]} for _,locs in repeated for k,h,span in locs]}
    (HERE/'full_search_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='closest_rmse_per_stratum_family'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
