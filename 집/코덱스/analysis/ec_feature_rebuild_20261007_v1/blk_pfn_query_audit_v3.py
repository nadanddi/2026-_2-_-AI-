"""Frozen-context audit of minbatch8 including all small batch sizes.

Same model, context, features and weight as v1. No heldout labels or scores.
Each context writes its own immutable evidence file.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
template = HERE / 'blk_pfn_query_audit_v1.py'
source = template.read_text(encoding='utf-8')
replacements = {
    'from checkpoint_v1 import atomic,digest':
        'from checkpoint_v1 import atomic,digest\nfrom blk_pfn_minbatch_v1 import predict_minbatch8',
    "out=HERE/f'BLK_PFN_context{seed}_query_audit_v1.json'":
        "out=HERE/f'BLK_PFN_context{seed}_query_audit_v3.json'",
    "'code_sha256':sha(__file__)":
        "'code_sha256':sha(__file__),'source_template_sha256':sha(HERE/'blk_pfn_query_audit_v1.py'),'adapter_sha256':sha(HERE/'blk_pfn_minbatch_v1.py'),'policy':'minimum8 query rows; own last-row copies only; unchanged fit'",
    "        subset=np.asarray(m.predict(Q[indices]),float)":
        "        print(f'context{seed}: fit complete; scattered96 start',flush=True)\n        subset=np.asarray(m.predict(Q[indices]),float)",
}
for old, new in replacements.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
assert source.count('m.predict(') == 4
source = source.replace('m.predict(', 'predict_minbatch8(m, ')
needle = '        # Protect earliest F13 query row.'
extra = """        print(f'context{seed}: order and last single complete; independent singles start',flush=True)
        singles=[]
        for position in order:
            value=float(predict_minbatch8(m,Q[[indices[position]]])[0])
            singles.append(abs(value-expected[position]))
        differences['single8_each_vs_original1440']=max(singles)
        for count in range(2,8):
            small_indices=np.asarray(indices)[order[:count]]
            small_expected=expected[order[:count]]
            result=predict_minbatch8(m,Q[small_indices])
            reverse_result=predict_minbatch8(m,Q[small_indices[::-1]])[::-1]
            differences[f'small{count}_vs_original1440']=float(np.max(np.abs(result-small_expected)))
            differences[f'small{count}_reversed_vs_original1440']=float(np.max(np.abs(reverse_result-small_expected)))
            print(f'context{seed}: small{count} checked',flush=True)
"""
assert source.count(needle) == 1
source = source.replace(needle, extra + needle)
source = source.replace(
    'all8 blocks, front/middle/back, hours0/5/12/23; order8, single1, poisonedbatch1. Finite empirical checks, not exhaustive input-domain proof',
    'all8 blocks, front/middle/back, hours0/5/12/23; order8, independent single8 plus last single, batch sizes2..7 forward/reverse, poisonedbatch1. Finite checks, not exhaustive input-domain proof')
exec(compile(source, str(Path(__file__).resolve()), 'exec'))
