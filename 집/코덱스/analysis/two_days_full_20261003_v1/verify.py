from extract import *
import csv
from html.parser import HTMLParser

def same(a,b):return (a is None and b is None) or (a is not None and b is not None and a==b)
def number(v):return None if v=='' else float(v)

class Reader(HTMLParser):
    def __init__(self):super().__init__();self.current=None;self.cells={};self.rows={};self.summary_rows=0
    def handle_starttag(self,tag,attrs):
        attr=dict(attrs)
        if tag=='details':self.current=attr['id'];self.cells[self.current]=0;self.rows[self.current]=0
        if tag=='tr' and self.current:self.rows[self.current]+=1
        if tag=='td' and self.current:self.cells[self.current]+=1
        if tag=='tr' and 'data-group' in attr:self.summary_rows+=1
    def handle_endtag(self,tag):
        if tag=='details':self.current=None

def main():
    d=json.loads((HERE/'data.json').read_text(encoding='utf-8'));ids={f'{f}_{day:03d}_{h:02d}' for f,day in CASES for h in range(24)}
    path=ROOT/'공用' if False else ROOT/'공용/대회자료/정형데이터/참가자_배포/train_X.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);inputcols=[c for c in reader.fieldnames if c!='row_id'];raw={r['row_id']:{c:number(r[c]) for c in inputcols} for r in reader if r['row_id'] in ids}
    assert len(raw)==48 and inputcols==d['raw_input_columns']
    oof=pd.read_csv(ROOT/'집/코덱스/local/temp_tk_season_20261003_v1/TK1_predictions.csv');oof=oof[(oof.validator=='DIAG10')&(oof.base_seed==7)&(oof.context=='1-8')&oof.row_id.isin(ids)]
    truth=oof[['row_id','sub_temp']].drop_duplicates().set_index('row_id').sub_temp.to_dict();assert len(truth)==48
    models={(r.row_id,r.member):float(r.prediction) for r in oof.itertuples() if r.member in ['BASE','CODEX','PFN','W30G']}
    e=pd.read_csv(ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv');e=e[(e.validator=='DIAG10')&(e.seed==7)&e.row_id.isin(ids)].set_index('row_id');assert len(e)==24
    table=pd.read_csv(HERE/'두날_전체원자료와예측_v1.csv');assert set(table.row_id)==ids and table.row_id.nunique()==48
    assert table.shape==(48,31)
    checked=0
    for r in table.to_dict(orient='records'):
        rid=r['row_id'];assert r['farm']==rid[:3] and r['day']==int(rid[4:7]) and r['hour']==int(rid[8:10])
        for col in inputcols:assert same(raw[rid][col],val(r[col]));checked+=1
        assert abs(r['sub_temp']-truth[rid])<1e-12;checked+=1
        for member in ['BASE','CODEX','PFN','W30G']:assert abs(r['temp_'+member]-models[(rid,member)])<1e-12;checked+=1
        assert abs(r['temp_error']-(models[(rid,'W30G')]-truth[rid]))<1e-12;checked+=1
        for out,col in [('sub_ec_public','sub_ec'),('ec_season_v2_public','season_v2')]:
            if rid in e.index:assert abs(r[out]-e.loc[rid,col])<1e-12
            else:assert pd.isna(r[out])
            checked+=1
    pairs=pd.read_csv(HERE/'시간별_항목별_대조_v1.csv');assert len(pairs)==648
    lookup=table.set_index('row_id');summary_checks=0
    for p in d['panels']:
        col=p['column'];left=[];right=[];stats=dict(same_hours=0,different_hours=0,both_missing_hours=0,one_missing_hours=0)
        for row in p['values']:
            hour=row['hour'];lid=f'F13_194_{hour:02d}';rid=f'F47_191_{hour:02d}';a=val(lookup.loc[lid,col]);b=val(lookup.loc[rid,col]);assert same(a,row['F13']) and same(b,row['F47'])
            if a is None and b is None:stats['both_missing_hours']+=1
            elif a is None or b is None:stats['one_missing_hours']+=1
            elif a==b:stats['same_hours']+=1
            else:stats['different_hours']+=1
            if a is not None:left.append(a)
            if b is not None:right.append(b)
            expected=b-a if a is not None and b is not None else None
            assert (expected is None and row['difference_F47_minus_F13'] is None) or abs(expected-row['difference_F47_minus_F13'])<1e-12
        for key,value in stats.items():assert p[key]==value
        for key,values in [('F13_mean',left),('F47_mean',right)]:
            if values:assert abs(math.fsum(values)/len(values)-p[key])<1e-10
            else:assert p[key] is None
        summary_checks+=1
    reader=Reader();reader.feed((HERE/'전체비교표_v1.html').read_text(encoding='utf-8'));assert len(reader.rows)==27 and reader.summary_rows==27;assert set(reader.rows.values())=={25} and set(reader.cells.values())=={96}
    inputs=[p for p in d['panels'] if p['group']=='입력'];equal=[p['column'] for p in inputs if p['same_hours']==24];different=[p['column'] for p in inputs if p['different_hours']>0];missing=[p['column'] for p in inputs if p['both_missing_hours']==24]
    assert (len(equal),len(different),len(missing))==(6,8,5)
    proof=dict(status='PASS',raw_value_checks=checked,summary_checks=summary_checks,hourly_pairs=len(pairs),html_panels=len(reader.rows),html_hours_per_panel=24,identical_input_columns=equal,different_input_columns=different,both_missing_input_columns=missing,ec_public_availability=dict(F13_194=0,F47_191=24),methods=['csv.DictReader raw inputs versus pandas extraction','published OOF labels/predictions','scalar equality/mean/difference versus vector comparison','HTMLParser table coverage'],hashes={p.name:sha(p) for p in HERE.glob('*') if p.is_file() and p.suffix in ['.csv','.html','.md']})
    (HERE/'verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(proof,ensure_ascii=False))

if __name__=='__main__':main()
