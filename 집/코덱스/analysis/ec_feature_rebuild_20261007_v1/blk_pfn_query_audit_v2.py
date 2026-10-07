"""Preserve v1, replay its exact probes with a pinned minbatch8 adapter.

This checks numerical repair; no performance scores or truth are loaded.
"""
from pathlib import Path
import hashlib

HERE = Path(__file__).resolve().parent
original = HERE / 'blk_pfn_query_audit_v1.py'
adapter = HERE / 'blk_pfn_minbatch_v1.py'
source = original.read_text(encoding='utf-8')
source = source.replace('from checkpoint_v1 import atomic,digest',
    'from checkpoint_v1 import atomic,digest\nfrom blk_pfn_minbatch_v1 import predict_minbatch8')
source = source.replace('m.predict(', 'predict_minbatch8(m, ')
source = source.replace('query_audit_v1.json', 'query_audit_v2.json')
source = source.replace("'code_sha256':sha(__file__)",
    "'code_sha256':sha(__file__),'source_template_sha256':sha(HERE/'blk_pfn_query_audit_v1.py'),'adapter_sha256':sha(HERE/'blk_pfn_minbatch_v1.py'),'policy':'minimum8 inference rows; copies of current last row only; no fit changes'")
# Each of the eight order probes gets an independent one-row call. This includes
# both farms, all eight blocks, and the previously failed last query row.
needle = "        # Protect earliest F13 query row."
extra = """        singles=[]
        for position in order:
            value=float(predict_minbatch8(m,Q[[indices[position]]])[0])
            singles.append(abs(value-expected[position]))
        differences['single8_each_vs_original1440']=max(singles)
"""
assert source.count(needle)==1
source = source.replace(needle, extra+needle)
assert 'm.predict(' not in source
exec(compile(source, str(Path(__file__).resolve()), 'exec'))
