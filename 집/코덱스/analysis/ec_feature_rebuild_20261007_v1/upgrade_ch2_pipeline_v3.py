from pathlib import Path
import ast
here=Path(__file__).resolve().parent
ctx=(here/'ch2_query_context_v1.py').read_text(encoding='utf-8')
old="""        values={}
        for other in sorted(expected):
            assert other in self._query and self.query_to_block[other]==index
            assert key(other)[0]==farm and key(other)[1:]<=(day,hour)
            raw=self._query[other]
            assert set(raw)==set(RAW)
            obs={c:float(raw[c]) if raw[c].strip() else None for c in RAW}
"""
new="""        # Validate every permitted row identity/schema before any .strip/float access.
        for other in sorted(expected):
            assert other in self._query and self.query_to_block[other]==index
            assert key(other)[0]==farm and key(other)[1:]<=(day,hour)
            assert set(self._query[other])==set(RAW)
        values={}
        for other in sorted(expected):
            raw=self._query[other]
            obs={c:float(raw[c]) if raw[c].strip() else None for c in RAW}
"""
assert old in ctx;ctx=ctx.replace(old,new)
ast.parse(ctx);out=here/'ch2_query_context_v2.py';assert not out.exists();out.write_text(ctx,encoding='utf-8')
runner=(here/'run_ch2_BLK_v2.py').read_text(encoding='utf-8')
runner=runner.replace('CH2_BLK_registration_v2.json','CH2_BLK_registration_v3.json')
runner=runner.replace('from ch2_query_context_v1 import CH2QueryContext','import ch2_query_context_v2 as context_module\nimport blk_context_v1 as base_context_module\nCH2QueryContext=context_module.CH2QueryContext')
marker="    registration=read('CH2_BLK_registration_v3.json')"
runner=runner.replace(marker,"""    assert sys.version_info[:2]==(3,12), 'registered Python3.12 runtime required'
    for module,name in [(context_module,'ch2_query_context_v2.py'),(base_context_module,'blk_context_v1.py'),
                        (reference_loader,'ch2_reference_loader_v1.py'),(prefix_module,'ch2_prefix_sources_v3.py')]:
        assert Path(module.__file__).resolve()==(HERE/name).resolve()
"""+marker)
ast.parse(runner);out=here/'run_ch2_BLK_v3.py';assert not out.exists();out.write_text(runner,encoding='utf-8')
reg=(here/'register_ch2_BLK_v2.py').read_text(encoding='utf-8')
reg=reg.replace('register_ch2_BLK_v2.py','register_ch2_BLK_v3.py').replace('run_ch2_BLK_v2','run_ch2_BLK_v3')
reg=reg.replace('ch2_query_context_v1.py','ch2_query_context_v2.py').replace('CH2_BLK_registration_v2.json','CH2_BLK_registration_v3.json').replace('feature_candidates_v7.csv','feature_candidates_v8.csv')
reg=reg.replace("'CH2_training_reference_preparation_v1.json','CH2_training_reference_file_replay_v1.json',", "'CH2_training_reference_preparation_v1.json','CH2_training_reference_file_replay_v1.json',\n                 'audit_ch2_query_context_synthetic_v1.py','CH2_query_context_synthetic_audit_v1.json',")
ast.parse(reg);out=here/'register_ch2_BLK_v3.py';assert not out.exists();out.write_text(reg,encoding='utf-8')
verify=(here/'verify_ch2_BLK_v1.py').read_text(encoding='utf-8').replace('run_ch2_BLK_v2','run_ch2_BLK_v3').replace('ch2_query_context_v1','ch2_query_context_v2').replace('CH2_BLK_registration_v2.json','CH2_BLK_registration_v3.json').replace('CH2_BLK_verified_gate_v1.json','CH2_BLK_verified_gate_v2.json')
ast.parse(verify);out=here/'verify_ch2_BLK_v2.py';assert not out.exists();out.write_text(verify,encoding='utf-8')
print('Created context2/runner3/registrar3/verifier2; earlier versions preserved')
