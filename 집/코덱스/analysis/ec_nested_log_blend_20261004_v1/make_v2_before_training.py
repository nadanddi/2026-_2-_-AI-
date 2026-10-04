"""Preserve executed preparation v1; add provenance/first-cell resume guards."""
from pathlib import Path
H = Path(__file__).resolve().parent
src = (H / 'run.py').read_text(encoding='utf-8')
dest = H / 'run_v2.py'
assert not dest.exists()
src = src.replace("def quant(x):", '''def features_sha(frame, columns):
    import hashlib
    x = np.asarray(frame[columns], dtype=np.float64).copy(order='C')
    x[np.isnan(x)] = np.nan
    header = json.dumps(dict(columns=columns, shape=x.shape), sort_keys=True).encode()
    return hashlib.sha256(header + x.tobytes(order='C')).hexdigest()

def first_artifact_guard(dest, meta_path, inner_path, signature):
    first = H / 'first_fold_verification_v2.json'
    files = [dest, meta_path, inner_path, first]
    present = [p.exists() for p in files]
    assert not any(present) or all(present), 'partial first cell/audit; stop before refit and preserve files'
    if all(present):
        record = json.loads(first.read_text(encoding='utf-8'))
        assert record['status'] == 'PASS' and record['signature'] == signature

def quant(x):''')
src = src.replace("check.update(run_source_sha256=S.sha(Path(__file__)), log_source_sha256=M_SHA,", '''deps = dict(core=S.sha(Path(core.__file__)),
                season=S.sha(Path(sys.modules[S.mapping.__module__].__file__)),
                env=S.sha(Path(env.__file__)), support=S_SHA, guard=N_SHA, log=M_SHA)
    input_sha = S.sha(Path(env.DATA) / 'train_X.csv')
    for v, k, _, _ in folds:
        for seed in SEEDS:
            original_meta = json.loads((N.ORIGINAL / f'{v}_{k}_r3_{seed}.json').read_text(encoding='utf-8'))
            assert original_meta['provenance']['shared']['input_sha256']['train_X.csv'] == input_sha
    public_cache_sha = S.sha(N.ORIGINAL / 'v2_integration_oof.csv')
    check.update(dependency_sha256=deps, train_input_sha256=input_sha, public_cache_sha256=public_cache_sha)
    check.update(run_source_sha256=S.sha(Path(__file__)), log_source_sha256=M_SHA,''')
src = src.replace("H / 'preparation_v1.json'", "H / 'preparation_v2.json'")
src = src.replace("inner_train_ids=vector_id_sha(a.row_id), inner_query_ids=vector_id_sha(b.row_id),", '''dependency_sha256=deps, train_input_sha256=input_sha, public_cache_sha256=public_cache_sha,
                             inner_train_features_sha256=features_sha(a, cols),
                             inner_query_features_sha256=features_sha(b, cols),
                             outer_train_features_sha256=features_sha(tr, cols),
                             outer_query_features_sha256=features_sha(q, cols),
                             inner_train_ids=vector_id_sha(a.row_id), inner_query_ids=vector_id_sha(b.row_id),''')
src = src.replace("if any(p.exists() for p in [dest, meta_path, inner_path]):", '''if v == 'DIAG10' and k == 0 and seed == 7:
                first_artifact_guard(dest, meta_path, inner_path, signature)
            if any(p.exists() for p in [dest, meta_path, inner_path]):''')
src = src.replace("S.savej(H / 'first_fold_verification_v1.json', dict(first, weight_fit=stats))", "first = dict(first, weight_fit=stats, signature=signature)")
src = src.replace("audits.append(validate_saved(dest, meta_path, inner_path, signature, ref, q))\n            print(v, k, seed, 'done weight'", '''audits.append(validate_saved(dest, meta_path, inner_path, signature, ref, q))
            if v == 'DIAG10' and k == 0 and seed == 7:
                S.savej(H / 'first_fold_verification_v2.json', first)
            print(v, k, seed, 'done weight' ''')
compile(src, str(dest), 'exec')
dest.write_text(src, encoding='utf-8')
verify_src = (H / 'verify_full_v1.py').read_text(encoding='utf-8')
verify_dest = H / 'verify_full_v2.py'
assert not verify_dest.exists()
verify_src = verify_src.replace("source_sha = S.sha(eh/'run.py')", "source_sha = S.sha(eh/'run_v2.py')")
compile(verify_src, str(verify_dest), 'exec')
verify_dest.write_text(verify_src, encoding='utf-8')
print('v2 sources created; no fit')
