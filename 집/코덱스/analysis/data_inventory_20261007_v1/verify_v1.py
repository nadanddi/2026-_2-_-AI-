from pathlib import Path
import sys, csv, json, collections, math, zipfile, io
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import pandas as pd
import numpy as np
OUT=Path(__file__).parent
def save(name,value):
    p=OUT/name
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def competition():
    data=Path(env.DATA);pandas_frames={};checks=[];manual={}
    for filename in ['train_X.csv','train_y.csv','test_X.csv','sample_submission.csv']:
        with (data/filename).open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        df=pd.read_csv(data/filename);pandas_frames[filename]=df
        m={'rows':len(rows),'unique_ids':len({r['row_id'] for r in rows}),'missing':{c:sum(r[c]=='' for r in rows) for c in df}}
        assert m['rows']==len(df) and m['unique_ids']==df.row_id.nunique()
        assert m['missing']==df.isna().sum().to_dict()
        for c in ['sub_ec','sub_temp']:
            if c in df:
                v=[float(r[c]) for r in rows if r[c]!='']
                assert len(v)==df[c].notna().sum()
                if v:
                    assert min(v)==df[c].min() and max(v)==df[c].max()
                    m[c]={'n':len(v),'min':min(v),'max':max(v),'zero':sum(x==0 for x in v),'negative':sum(x<0 for x in v),'mean':math.fsum(v)/len(v)}
        manual[filename]=m;checks.append(filename+' row/id/missing/range independent PASS')
    x,y,t,s=(pandas_frames[k] for k in ['train_X.csv','train_y.csv','test_X.csv','sample_submission.csv'])
    joined=x.merge(y,on='row_id',how='left',validate='one_to_one',indicator=True)
    assert len(joined)==len(x)
    joined['farm']=joined.row_id.str.split('_').str[0]
    joined['day']=joined.row_id.str.split('_').str[1].astype(int)
    joined['hour']=joined.row_id.str.split('_').str[2].astype(int)
    farms=[];usable=['out_temp','out_hum','out_rad','out_wspd','in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
    for farm,g in joined.groupby('farm'):
        ec=g.sub_ec.notna();target=g.sub_temp.notna()
        farms.append({'farm':farm,'input_rows':len(g),'farm_days':g.day.nunique(),'day_min':g.day.min(),'day_max':g.day.max(),'ec_labels':int(ec.sum()),'temp_labels':int(target.sum()),'no_label_id':int(g['_merge'].eq('left_only').sum()),'ec_min':g.sub_ec.min() if ec.any() else None,'ec_max':g.sub_ec.max() if ec.any() else None,'missing_usable':g[usable].isna().sum().to_dict(),'complete14_rows':int(g[usable].notna().all(axis=1).sum())})
    target=joined[joined.farm.isin(['F13','F47'])];labeled=target[target.sub_ec.notna()]
    anomalies={}
    for c in usable:
        vals=target[c];p={'missing':int(vals.isna().sum()),'negative':int(vals.lt(0).sum()),'zero':int(vals.eq(0).sum()),'nonfinite':int((vals.notna()&~np.isfinite(vals)).sum())}
        if c in ['out_hum','in_hum'] or c.startswith('act_'):p['outside_0_100']=int((vals.lt(0)|vals.gt(100)).sum())
        anomalies[c]=p
    counts=labeled.groupby(['farm','day']).hour.nunique()
    testcoord=t.row_id.str.extract(r'^(.*)_(\d+)_(\d+)$');testcoord.columns=['farm','day','hour'];testcoord[['day','hour']]=testcoord[['day','hour']].astype(int)
    coverage=[]
    for farm,g in target.groupby('farm'):
        train_days=set(g.day);test_days=set(testcoord[testcoord.farm.eq(farm)].day);all_days=train_days|test_days
        coverage.append({'farm':farm,'train_days':len(train_days),'test_days':len(test_days),'same_day_overlap':len(train_days&test_days),'range':[min(all_days),max(all_days)],'missing_day_numbers_in_range':sorted(set(range(min(all_days),max(all_days)+1))-all_days)})
    save('competition_join_and_independent_v1.json',{'independent_checks':checks,'manual':manual,'farms':farms,'ec_scope':{'input_rows':len(target),'labeled_rows':len(labeled),'unlabeled_rows':len(target)-len(labeled),'labeled_farm_days':len(counts),'incomplete_24h_labeled_days':int(counts.ne(24).sum()),'all14_complete_rows':int(labeled[usable].notna().all(axis=1).sum())},'target_input_anomalies':anomalies,'train_test_day_coverage':coverage,'id_join':'one_to_one left join; no rows cleaned','mask_rule':'5 test-all-NaN columns excluded from train features as well; profiling does not impute/drop rows'})
    print('COMPETITION_VERIFIED',len(farms),'farms',len(labeled),'EC labels',flush=True)
def external_verify():
    verified=[];raw=ROOT/'집/코덱스/local/smartfarm_all_structured_20261004_v1'
    for path in sorted(OUT.glob('external_*_v1.json')):
        if path.name=='external_manifest_v1.json':continue
        d=json.loads(path.read_text(encoding='utf-8'));p=raw/d['file']
        if p.suffix!='.zip':continue
        with zipfile.ZipFile(p) as z:
            for m in d['members']:
                b=z.read(m['member']);txt=b.decode(m['encoding']);rows=csv.DictReader(io.StringIO(txt));n=0;sf=0;groups=collections.defaultdict(lambda:collections.Counter());farms=set();sensor=m['sensor']
                for r in rows:
                    n+=1;crop=sensor['crop_column']
                    if crop and not ('딸기' in r[crop] or r[crop] in ['80400','080400']):continue
                    sf+=1
                    if sensor['farm_column']:farms.add(r[sensor['farm_column']])
                    if sensor['item_column'] and sensor['value_column']:
                        g=groups[r[sensor['item_column']]];g['n']+=1
                        try:
                            v=float(r[sensor['value_column']])
                            if math.isfinite(v):g['zero']+=v==0;g['negative']+=v<0
                        except ValueError:pass
                assert n==m['rows']
                if crop:assert sf==sensor['strawberry_rows']
                if sensor['farm_column']:assert len(farms)==sensor['farms']
                for g in sensor['groups']:
                    c=groups[g['item']];assert all(c[k]==g[k] for k in ['n','zero','negative']), (d['file'],g['item'])
                verified.append({'file':d['file'],'member':m['member'],'rows':n,'strawberry_rows':sf if crop else None,'groups_checked':len(groups),'independent_csv_pass':True})
        print('EXTERNAL_VERIFIED',d['file'],flush=True)
    save('external_independent_v1.json',verified)
if __name__=='__main__':globals()[sys.argv[1]]()
