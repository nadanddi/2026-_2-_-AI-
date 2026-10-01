"""Fresh verification of descriptive report values; no new search or tuning."""
from pathlib import Path
import csv,json,math
from collections import Counter
HERE=Path(__file__).resolve().parent
def read(name):
    with (HERE/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    s=json.loads((HERE/'summary.json').read_text(encoding='utf-8'))
    v=json.loads((HERE/'full_search_verification.json').read_text(encoding='utf-8'))
    null=read('null_counts.csv')
    background={}
    for family in s['families']:
        totals=[sum(int(r[f'{stratum}/{family}']) for stratum in s['strata']) for r in null]
        background[family]={'surrogates_with_any_match':sum(t>0 for t in totals),'total_surrogates':len(totals),
                            'mean_independent_pairs':math.fsum(totals)/len(totals),'max_independent_pairs':max(totals)}
    plateau=read('plateau_runs.csv')
    ds={}
    for r in read('days.csv'):
        st=r['stratum'];ds.setdefault(st,{'days':0,'informative_days':0,'min_record_day':100000,'max_record_day':0})
        ds[st]['days']+=1;ds[st]['informative_days']+=int(r['informative']=='True')
        ds[st]['min_record_day']=min(ds[st]['min_record_day'],int(r['day']))
        ds[st]['max_record_day']=max(ds[st]['max_record_day'],int(r['day']))
    locs=v['repeated_window_days'];gap=abs(locs[0]['mean']-locs[1]['mean'])
    assert len(plateau)==214 and sum(d['informative_days'] for d in ds.values())==270
    closest=min(q['rmse'] for q in v['closest_rmse_per_stratum_family'].values())
    r={'status':'PASS','null_background':background,'strata_counts':ds,'primary_eligible_day_pairs':sum(v['primary_eligible_pairs_by_stratum'].values()),
       'closest_primary_match_rmse':closest,'closest_to_match_threshold_ratio':closest/.0005,
       'plateau_runs':len(plateau),'plateau_hours':sum(int(z['length']) for z in plateau),
       'plateau_days':len({(z['farm'],z['day']) for z in plateau}),'plateau_max_length':max(int(z['length']) for z in plateau),
       'repeated_window_daymean_gap':gap,'repeated_gap_to_v2_level_rmse_ratio':gap/s['v2_day_level_rmse'],
       'otherfarm_nonempty_ec_count':sum(q['ec_nonempty'] for f,q in s['otherfarm_label_inventory'].items() if f not in ['F13','F47']),
       'otherfarm_rows':sum(q['rows'] for f,q in s['otherfarm_label_inventory'].items() if f not in ['F13','F47']),
       'otherfarm_count':len(s['otherfarm_label_inventory'])-2}
    (HERE/'final_gate.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
