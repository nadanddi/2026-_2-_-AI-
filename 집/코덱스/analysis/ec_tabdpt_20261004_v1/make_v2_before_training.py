"""Preserve prepared v1; apply pre-fit review guards without changing family20 math."""
from pathlib import Path
H=Path(__file__).resolve().parent
target=H/'run_v2.py'
assert not target.exists()
s=(H/'run_v1.py').read_text(encoding='utf-8')
needle="def array_sha(x):"
insert="""def compare_nullable(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    assert a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b))
    assert not np.isinf(a).any() and not np.isinf(b).any()
    mask=np.isfinite(a)
    return compare(a[mask],b[mask]) if mask.any() else 0.
def dependency_hashes(core):
    return dict(core=S.sha(Path(core.__file__)),season=S.sha(Path(sys.modules[S.mapping.__module__].__file__)),
                env=S.sha(Path(env.__file__)),support=S_SHA,adapter=S.sha(PREP/'adapter_draft_v1.py'),
                runtime_probe=S.sha(PREP/'runtime_probe_v3.py'),run=S.sha(Path(__file__)))
def validate_first(record,signature):
    assert record['status']=='PASS' and record['signature']==signature
    assert record['raw_atol']==RAW_ATOL and record['final_atol']==FINAL_ATOL
    assert set(record['raw_errors'])=={'repeat','single','reversed','other_query','fresh_fit'}
    assert all(np.isfinite(x) and 0<=x<=RAW_ATOL for x in record['raw_errors'].values())
    assert np.isfinite(record['scalar_maxdiff']) and 0<=record['scalar_maxdiff']<=1e-12
    expected={(f,h) for f in ['F13','F47'] for h in [0,6,12]}
    assert len(record['prefix'])==6 and {(x['farm'],x['hour']) for x in record['prefix']}==expected
    assert all(0<=x['raw_maxdiff']<=RAW_ATOL and 0<=x['final_maxdiff']<=FINAL_ATOL for x in record['prefix'])
    assert len(record['feature_causal'])==6 and {(x['farm'],x['hour']) for x in record['feature_causal']}==expected
    assert all(0<=x['feature_maxdiff']<=1e-12 for x in record['feature_causal'])
"""
assert s.count(needle)==1
s=s.replace(needle,insert+needle)
needle="    runtime=runtime_core(); refs={}; manifests=[]"
assert s.count(needle)==1
s=s.replace(needle,needle+"\n    deps=dependency_hashes(core); input_sha=S.sha(Path(env.DATA)/'train_X.csv'); public_sha=S.sha(OLD/'v2_integration_oof.csv')")
needle="            r3=dict(np.load(OLD/f'{v}_{k}_r3_{seed}.npz',allow_pickle=False))"
assert s.count(needle)==1
s=s.replace(needle,"""            original_meta=json.loads((OLD/f'{v}_{k}_r3_{seed}.json').read_text(encoding='utf-8'))
            assert original_meta['provenance']['shared']['input_sha256']['train_X.csv']==input_sha
            assert original_meta['provenance']['shared']['core_sha256']==deps['core']
            assert all(original_meta['provenance']['environment'][name]==value for name,value in runtime.items())
"""+needle)
needle="                 train_ids=ids_sha(tr.row_id),query_ids=ids_sha(q.row_id),"
assert s.count(needle)==1
s=s.replace(needle,"                 dependencies=deps,input_sha256=input_sha,public_cache_sha256=public_sha,\n"+needle)
needle="dict(status='PASS',folds=22,manifest=manifests,runtime=runtime,fit_count=0)"
assert s.count(needle)==1
s=s.replace(needle,"dict(status='PASS',folds=22,manifest=manifests,runtime=runtime,fit_count=0,dependencies=deps,input_sha256=input_sha,public_cache_sha256=public_sha)")
s=s.replace("gap=compare(original.loc[ids,cols[:-1]],new.loc[ids,cols[:-1]])","gap=compare_nullable(original.loc[ids,cols[:-1]],new.loc[ids,cols[:-1]])")
s=s.replace("S.savej(H/'preparation_v1.json',prepared)","S.savej(H/'preparation_v2.json',prepared)")
needle="    assert not (H/'fit_audit_v1.json').exists(),'Complete experiment: verify without retraining.'"
assert s.count(needle)==1
s=s.replace(needle,"    assert json.loads((H/'preparation_v2.json').read_text(encoding='utf-8'))==prepared,'Prepared inputs/code changed: stop before fitting.'\n"+needle)
needle="                saved=json.loads(meta.read_text(encoding='utf-8'));assert saved['signature']==signature and saved['csv_sha256']==S.sha(dest)"
assert s.count(needle)==1
s=s.replace(needle,needle+" and saved['status']=='PASS'")
s=s.replace("if len(files)==3:assert json.loads(first.read_text(encoding='utf-8'))['signature']==signature","if len(files)==3:validate_first(json.loads(first.read_text(encoding='utf-8')),signature)")
needle="                compare(d.y,q.sub_ec);compare(d.baseline,old[f'baseline_{seed}'])"
assert s.count(needle)==1
s=s.replace(needle,needle+"\n                for c in ['farm','day','hour']:assert np.array_equal(d[c],q[c])\n                assert (d.validator==v).all() and (d.fold==k).all() and (d.seed==seed).all()\n                compare(d.r3_raw,old[f'r3_{seed}']);compare(d.old_pfn_raw,old['old_pfn_raw'])\n                compare(d.clip_lo,np.repeat(old['lo'],len(q)));compare(d.clip_hi,np.repeat(old['hi'],len(q)))")
needle="                if check is not None:S.savej(first,dict(**check,signature=signature))"
assert s.count(needle)==1
s=s.replace(needle,"                if check is not None:\n                    first_record=dict(**check,signature=signature);validate_first(first_record,signature);S.savej(first,first_record)")
target.write_text(s,encoding='utf-8')
print('V2_CREATED_NO_FIT')
