import sys
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,csv,tarfile,hashlib,collections,unicodedata
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local/image_source_mapping_20261010_v1'
obj=json.loads((RUN/'mapping_v1.json').read_text(encoding='utf-8'))
labels=list(csv.DictReader((RUN.parent/'image_download_complete_20261010_v1/validation_metadata_v1.csv').open(encoding='utf-8-sig')))
normal=lambda s:unicodedata.normalize('NFC',s).casefold()
checks=[];read=0
for a in obj['archives']:
    rows={normal(PurePosixPath(r['member']).stem):r for r in labels if r['archive']=='VL'+a['name'][2:]}
    mapped={normal(PurePosixPath(r['source_member']).stem):r for r in obj['mapped'] if r['source_archive']==a['name']}
    with tarfile.open(LOCAL/a['name'],'r:') as tar:
        members=[m for m in tar.getmembers() if m.isfile()]
        assert len(members)==len(rows)==len(mapped)==500
        assert {normal(PurePosixPath(m.name).stem) for m in members}==set(rows)==set(mapped)
        for m in members:
            stem=normal(PurePosixPath(m.name).stem);r=mapped[stem];label=rows[stem]
            assert r['image_id']==label['image_id'] and r['farm_id']==label['farm_id'] and r['member']==label['member']
            assert r['offset']==m.offset_data and r['bytes']==m.size
            with tar.extractfile(m) as f:b=f.read(32*1024**2+1)
            read+=len(b)
            assert len(b)==m.size and hashlib.sha256(b).hexdigest()==r['payload_sha256']
    checks.append({'archive':a['name'],'rows':len(rows),'all_offsets_sizes_hashes_label_keys_match':True})
groups=collections.defaultdict(list)
for r in obj['mapped']:groups[r['payload_sha256']].append(r)
duplicates=[{'sha256':h,'rows':rr,'lineage_rule':'all exact matches remain in one connected component; preserve rows'} for h,rr in groups.items() if len(rr)>1]
assert len(duplicates)==obj['within_VS_duplicate_sha_groups']
components=[]
# Farm+candidate prefix forms a conservative source-lineage grouping; not authenticated plant IDs.
parent={str(i):str(i) for i in range(len(obj['mapped']))}
def find(i):
    while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
    return i
def union(a,b):parent[find(b)]=find(a)
bygroup={};byhash={}
for i,r in enumerate(obj['mapped']):
    idx=str(i);key=(r['farm_id'],r['kind_type'],r['prefix_candidate']);h=r['payload_sha256']
    if key in bygroup:union(idx,bygroup[key])
    else:bygroup[key]=idx
    if h in byhash:union(idx,byhash[h])
    else:byhash[h]=idx
for i,r in enumerate(obj['mapped']):components.append({'source_archive':r['source_archive'],'source_member':r['source_member'],'image_id':r['image_id'],'farm_id':r['farm_id'],'component':'source-component-'+find(str(i)),'assignment':'development_only; derived sources/donors must inherit same component'})
reservation=json.loads((RUN.parent/'image_ingestion_20261010_v1/split_reservation_v1.json').read_text(encoding='utf-8'));folds=collections.defaultdict(set)
for r in components:folds[r['component']].add(reservation['development_farm_folds'][r['farm_id']])
conflicts={k:sorted(v) for k,v in folds.items() if len(v)>1}
result={'cross_fold_component_conflicts':conflicts,'checks':checks,'independent_method':'stdlib tarfile against CSV labels, offsets/size and payload SHA freshly reread','payload_read_bytes':read,'mapped_rows':len(obj['mapped']),'exact_duplicate_groups':len(duplicates),'exact_duplicate_rows':sum(len(x['rows']) for x in duplicates),'source_components':len({r['component'] for r in components}),'scope':'1000 VS and cached1000 TS exact-hash comparison; no full-data or pixel-equivalence guarantee'}
for name,value in [('verification_v1.json',result),('exact_duplicates_v1.json',duplicates),('source_lineage_v1.json',components)]:
    with (RUN/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
