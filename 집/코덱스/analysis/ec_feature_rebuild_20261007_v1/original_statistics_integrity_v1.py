"""Draw integrity only, explicitly NOT a full-model score gate or truth loader."""
from pathlib import Path
import hashlib,json,struct,math,platform
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def validate_draw_file(path,expected_sha,layout,draw_count):
    keys=layout['block_keys'];groups=layout['groups']
    n=layout['rows_per_block'];nt=layout['TM_rows_per_block']
    assert len(keys)>0 and len(keys)==len(n)==len(nt) and draw_count>0
    assert len({tuple(k) for k in keys})==len(keys)
    assert all(isinstance(v,int) and v>0 for v in n)
    assert all(isinstance(v,int) and 0<=v<=a for v,a in zip(nt,n)) and sum(nt)>0
    assert groups and all(groups)
    flat=[i for g in groups for i in g]
    assert sorted(flat)==list(range(len(keys)))
    assert all(len({keys[i][0] for i in g})==1 for g in groups)
    width=len(keys)*2;path=Path(path)
    assert path.stat().st_size==draw_count*width
    assert sha(path)==expected_sha
    fmt='<'+str(len(keys))+'H'
    with path.open('rb') as h:
        for _ in range(draw_count):
            raw=h.read(width);assert len(raw)==width
            counts=struct.unpack(fmt,raw)
            assert all(sum(counts[i] for i in g)==len(g) for g in groups)
            assert sum(c*v for c,v in zip(counts,n))>0
            assert sum(c*v for c,v in zip(counts,nt))>0
        assert h.read(1)==b''
    return draw_count

def validate_sealed_statistics():
    assert platform.python_version().startswith('3.12.')
    path=HERE/'DOMAIN24_original_statistics_registration_v2.json'
    reg=json.loads(path.read_text(encoding='utf-8'))
    assert reg['status']=='ORIGINAL_DOMAIN_DIAGNOSTIC_DRAWS_SEALED_BEFORE_FIT_NO_LABELS'
    assert all(sha(p)==v for p,v in reg['source_sha256'].items())
    assert canonical(reg['layout'])==reg['layout_sha256']
    assert reg['draws']==200000 and reg['random_seed']==2026100703
    assert reg['comparisons']==84 and reg['alpha']==.025/84
    assert len(reg['layout']['ordered_DIAG_ids'])==8640 and len(reg['layout']['ordered_TM_ids'])==2664
    count=validate_draw_file(HERE/'DOMAIN24_original_bootstrap_draws_v2.bin',reg['draw_file_sha256'],reg['layout'],200000)
    assert all(sha(p)==v for p,v in reg['source_sha256'].items())
    return {'statistics_registration_sha256':sha(path),'draw_sha256':reg['draw_file_sha256'],
            'draws_checked':count,'python_actual':platform.python_version(),
            'full_model_score_gate':False,'heldout_truth_loaded':False}

if __name__=='__main__':
    result=validate_sealed_statistics()
    result.update(status='STATISTICS_INTEGRITY_PASS_NOT_FULL_MODEL_GATE',code_sha256=sha(__file__))
    with (HERE/'DOMAIN24_original_statistics_integrity_v1.json').open('x',encoding='utf-8') as h:
        json.dump(result,h,ensure_ascii=False,indent=2)
    print('Draw/source/layout/runtime integrity PASS; no full model gate or truth access')
