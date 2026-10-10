import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
import csv,json,sqlite3,collections,hashlib
RUN=Path(__file__).resolve().parent
p=RUN/'metadata_all_v1.csv';rows=list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
profile=json.loads((RUN/'metadata_all_profile_v1.json').read_text(encoding='utf-8'))
db=sqlite3.connect(':memory:');db.execute('create table records(farm text,cultivar text,archive text,fname text,prefix text)')
db.executemany('insert into records values(?,?,?,?,?)',[(r['farm_id'],r['kind_type'],r['archive'],r['fname'],r['prefix_candidate']) for r in rows])
sql=[{'farm':a,'cultivar':b,'rows':n} for a,b,n in db.execute('select farm,cultivar,count(*) from records group by farm,cultivar order by farm,cultivar')]
checks={'rows_csv_equal_profile':len(rows)==profile['records'],'rows_sql_equal_csv':db.execute('select count(*) from records').fetchone()[0]==len(rows),'cross_sql_equal_profile':sql==profile['farm_cultivar_rows'],'archive_sum_equal_rows':sum(a['records'] for a in profile['archives'])==len(rows),'archive_bytes_sum_equal_read':sum(a['bytes'] for a in profile['archives'])==profile['direct_read_bytes'],'fname_duplicate_sql_equal':db.execute('select count(*)-count(distinct fname) from records').fetchone()[0]==profile['duplicate_fname_excess'],'sha_equal':hashlib.sha256(p.read_bytes()).hexdigest()==profile['snapshot_sha256']}
assert all(checks.values()),checks
summary={'checks':checks,'csv_rows':len(rows),'farms':db.execute('select count(distinct farm) from records').fetchone()[0],'cultivar_counts':dict(collections.Counter(r['kind_type'] for r in rows)),'archive_count':db.execute('select count(distinct archive) from records').fetchone()[0],'gold_farms':[r[0] for r in db.execute("select distinct farm from records where cultivar='금실'")],'farm_cultivar_rows':sql,'schema_metadata_only':True,'bbox_not_checked':True,'independent_scene_count_unknown':True}
(RUN/'verification_v1.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
inv=json.loads((RUN/'inventory_v1.json').read_text(encoding='utf-8'));available=[]
for r in inv['rows']:
    if r['kind']=='TS' and r['candidate']:
        tl=r['name'].replace('TS_','TL_',1);match=[x for x in rows if x['archive']==tl]
        available.append({'TS':r['name'],'listed_bytes':r['bytes'],'metadata_rows':len(match),'farm_cultivar':dict(collections.Counter(x['farm_id']+'|'+x['kind_type'] for x in match)),'prefix_candidate_count':len({x['prefix_candidate'] for x in match}),'source_payload_verified':False})
(RUN/'source_selection_metadata_v1.json').write_text(json.dumps(available,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False));print(json.dumps(available[:3]+[a for a in available if a['TS'].startswith('TS_2')][:3],ensure_ascii=False))
