import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
import csv,json,sqlite3,unicodedata
from collections import Counter
RUN=Path(__file__).resolve().parent;PREV=RUN.parent/'image_ingestion_20261010_v1'
profile=json.loads((RUN/'validation_full_v1.json').read_text(encoding='utf-8'))
val=list(csv.DictReader((RUN/'validation_metadata_v1.csv').open(encoding='utf-8-sig',newline='')))
train=list(csv.DictReader((PREV/'metadata_all_v1.csv').open(encoding='utf-8-sig',newline='')))
db=sqlite3.connect(':memory:')
for table,rows in [('tr',train),('va',val)]:
    db.execute('create table '+table+'(farm text,kind text,fname text,id text,prefix text,archive text)')
    db.executemany('insert into '+table+' values(?,?,?,?,?,?)',[(r['farm_id'],r['kind_type'],unicodedata.normalize('NFC',Path(r['fname']).name).casefold(),r['image_id'],r['prefix_candidate'],r['archive']) for r in rows])
q=lambda s:db.execute(s).fetchone()[0]
cross=[{'farm':farm,'cultivar':kind,'rows':n} for farm,kind,n in db.execute('select farm,kind,count(*) from va group by farm,kind order by farm,kind')]
checks={'csv_rows':len(val)==profile['rows'],'sql_rows':q('select count(*) from va')==profile['rows'],'farm_cultivar_sql':cross==profile['farm_cultivar'],'archive_json_row_sum':sum(a['JSON_rows'] for a in profile['archives'])==len(val),'fname_overlap_sql':q('select count(distinct va.fname) from va join tr on va.fname=tr.fname')==profile['cross_shared_normalized_fnames'],'id_overlap_sql':q('select count(distinct va.id) from va join tr on va.id=tr.id')==profile['cross_shared_global_ids'],'composite_id_overlap_sql':q('select count(*) from (select distinct va.farm,va.kind,va.id from va join tr on va.farm=tr.farm and va.kind=tr.kind and va.id=tr.id)')==profile['cross_shared_composite_ids'],'prefix_group_overlap_sql':q('select count(*) from (select distinct va.farm,va.kind,va.prefix from va join tr on va.farm=tr.farm and va.kind=tr.kind and va.prefix=tr.prefix)')==profile['cross_shared_prefix_candidate_groups'],'valid_identity_all':len(profile['invalid_identity'])==0}
assert all(checks.values()),checks
result={'checks':checks,'validation_rows':len(val),'training_rows':len(train),'total_label_rows':len(train)+len(val),'label_archives':len(profile['archives'])+271,'farm_cultivar_sql':cross,'within_val_fname_excess':q('select count(*)-count(distinct fname) from va'),'label_structure_errors':len(profile['issues']),'scope':'metadata cross-check only, image bytes not compared'}
(RUN/'verification_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
