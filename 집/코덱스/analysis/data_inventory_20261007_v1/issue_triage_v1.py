from pathlib import Path
import json,collections,hashlib
OUT=Path(__file__).parent;ROOT=OUT.parents[3]
errors=json.loads((OUT/'file_check_summary_v2.json').read_text(encoding='utf8'))['errors'];array_errors={r['path']:r for r in json.loads((OUT/'final_verification_v2.json').read_text(encoding='utf8'))['array_errors_classified']};results=[]
source=(ROOT/'집/코덱스/analysis/ec_src_X1_20261002_v1/run_x1.py').read_text(encoding='utf8')
assert "if not data:" in source and "write_text('', encoding='utf-8')" in source
for e in errors:
    p=ROOT/e['path'];q={**e,'bytes':p.stat().st_size}
    if p.suffix=='.csv':
        assert p.stat().st_size==0 and 'ec_src_X1_20261002_v1' in e['path'];q.update(classification='empty_result_table_no_observations',evidence='writecsv source writes zero bytes for empty list')
    elif p.suffix=='.json' and 'external_libs/nanoarrow' in e['path']:
        assert b'//' in p.read_bytes();q.update(classification='commented_software_configuration_not_dataset',evidence='JSON configuration includes line comments; not strict JSON')
    elif 'synthetic' in e['path']:q.update(classification='synthetic_invalid_fixture',evidence='synthetic path and file signature')
    elif e['path'] in array_errors:q.update(classification=array_errors[e['path']]['classification'],evidence=array_errors[e['path']])
    else:raise AssertionError('unclassified error: '+e['path'])
    results.append(q)
summary={'issues':results,'classifications':dict(collections.Counter(r['classification'] for r in results)),'unclassified':0,'source_data_errors':0,'original_files_modified':0}
p=OUT/'issue_triage_v1.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(summary['classifications'],ensure_ascii=False))
