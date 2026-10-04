from pathlib import Path
import ast
H=Path(__file__).resolve().parent
s=(H/'run_v1.py').read_text(encoding='utf-8-sig')
s=s.replace('preparation_v1.json','preparation_v2.json')
s=s.replace("def library_guard():", "def library_guard():")
s=s.replace("synthetic_sha256=S.sha(H/'synthetic_result_v4.json'))", "synthetic_sha256=S.sha(H/'synthetic_result_v4.json'),cpp_audit_source_sha256=S.sha(H/'first_cpp_audit_v2.py'))")
s=s.replace("    bind_library()", "    assert json.loads((H/'synthetic_result_v4.json').read_text())['status']=='PASS_SYNTHETIC_ONLY'\n    bind_library()")
s=s.replace("record['signature']==sig and record['npz_sha256']==S.sha(dest)","record['status']=='PASS' and record['seed']==seed and record['signature']==sig and record['preparation_sha256']==prepared_hash and record['preregistration_sha256']==prereg_hash and record['npz_sha256']==S.sha(dest)")
s=s.replace("record=dict(status='PASS',signature=sig,seed=seed,npz_sha256", "record=dict(status='PASS',signature=sig,seed=seed,preparation_sha256=prepared_hash,preregistration_sha256=prereg_hash,npz_sha256")
s=s.replace("                model=make(tr,cols,seed);raw=predict(model,q,cols)",'''                trace=None
                if (v,k,seed)==('DIAG10',0,7):
                    trace=OUT/'first_exact_cpp_trace_v1.ndjson';assert not trace.exists();trace.open('x').close()
                    os.chdir(OUT);os.environ['FARMAI_LGB_AUDIT_PATH']=trace.name
                try:model=make(tr,cols,seed)
                finally:os.environ.pop('FARMAI_LGB_AUDIT_PATH',None)
                raw=predict(model,q,cols)
''')
s=s.replace("                    check=first_audit(", "                    check=first_audit(")
s=s.replace("                except BaseException:\n                    preserve_failed_raw('candidate_lgb'",'''                    if check is not None:
                        from first_cpp_audit_v2 import audit_trace
                        check['cpp_audit']=audit_trace(trace,model,tr,cols)
                        check['cpp_trace_sha256']=S.sha(trace)
                except BaseException:
                    preserve_failed_raw('candidate_lgb' ''').replace("preserve_failed_raw('candidate_lgb' ,", "preserve_failed_raw('candidate_lgb',")
s=s.replace("saved=dict(status='PASS',signature=signature,csv_sha256=S.sha(dest),", "saved=dict(status='PASS',signature=signature,first_audit_sha256=S.sha(first) if first.exists() else None,csv_sha256=S.sha(dest),")
s=s.replace("    assert record['status']=='PASS' and", "    assert record['cpp_audit']['status']=='PASS' and record['cpp_trace_sha256']==S.sha(OUT/'first_exact_cpp_trace_v1.ndjson')\n    assert record['status']=='PASS' and")
s=s.replace("    assert not list(OUT.glob('failure_*_raw.npz'))", "    assert not list(H.glob('failure_*.json')), 'A failed run is preserved: fresh reviewed version required'\n    assert not list(OUT.glob('failure_*_raw.npz'))")
ast.parse(s)
with (H/'run_v2.py').open('x',encoding='utf-8') as f:f.write(s)
print('RUNNER_V2_GENERATED_NO_FIT')
