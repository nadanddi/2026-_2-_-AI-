from pathlib import Path
import json,csv,hashlib,re,datetime
H=Path(__file__).resolve().parent;ROOT=H.parents[3];L=ROOT/'연구실/코덱스/local'/H.name
def js(n):return json.loads((H/n).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=js('implementation_manifest_v1.json')
for name,digest in manifest['files'].items():assert sha((ROOT if name.startswith('연구실/') else H)/name)==digest
receipt=js('receipt_v1.json');score=js('score_v1/completion.json');critic=js('critic_verify_results_v1.json');tree=js('tree_audit_v1.json');support=js('critic_verify_support_identity_v2.json')
assert not (L/'worker.lock').exists() and receipt['rows_sha']==score['input_sha']==sha(L/'rows.csv')
assert receipt['new_ET_fits']==30 and receipt['other_new_fit']==0 and receipt['source_sha']==sha(H/'run_v1.py') and score['source_sha']==sha(H/'score_v1.py')
assert critic['status']=='PASS_TWO_H0_ALL_CACHES_PIPELINE_SCORES_AND_BOOTSTRAP' and critic['rows']==69120 and critic['fit']==0
assert tree['status']=='PASS_FULL_600_TREE_PATHS_SUPPORT_AND_RETAINED_MEDIAN' and tree['nodes']==6821872 and tree['trace_sha']==sha(L/'trace.npz')
assert support['status']=='PASS_TRACE_LINK_SUPPORT_ARITHMETIC_AND_INPUT_IDENTITY'
assert not score['DIAG_EXPANSION_SCREEN'] and score['PASS2_ADOPTION_HOLD_FLAG'] and score['bootstrap']['worse_or_equal_count']==19287
for name,digest in score['files'].items():assert sha(H/'score_v1'/name)==digest
with (L/'rows.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==len({(r['arm'],r['seed'],r['row_id']) for r in rows})==69120
assert len({r['row_id'] for r in rows})==8640 and len({(r['farm'],r['day']) for r in rows})==360
scope=js('critic_input_imputation_scope_v1.json');a=[r['features']['in_co2']['current_median'] for r in scope['fold_medians']];b=[r['features']['in_co2']['h0_median'] for r in scope['fold_medians']];assert len(a)==10 and (min(a),max(a),min(b),max(b))==(403,408,410,416)
report=H/'실험결과_v4.md';text=report.read_text(encoding='utf-8-sig');links=re.findall(r'\]\((C:/work/[^)]+)\)',text);assert len(links)==9
for path in links:assert Path(path).exists()
assert '정리 중인 초안' not in text
files=['실험결과_v4.md','보완기록_v1.md','최종혹독비평_v1.md','최종혹독비평_보고서대조_v2.md','receipt_v1.json','score_v1/completion.json','critic_verify_results_v1.json','tree_audit_v1.json','critic_verify_support_identity_v2.json','critic_input_imputation_scope_v1.json']
files.extend(['case_chain_v1.json','case_chain_v1.csv','critic_case_chain_v1.json','최종혹독비평_보고서대조_v3.md'])
out=dict(status='COMPLETE_TWO_H0_DIAGNOSTIC_SCREEN_REJECT',completed_at=datetime.datetime.now().isoformat(),source_registration='7f13d0e',rows=69120,unique_query_rows=8640,unique_days=360,new_ET_fit=30,new_other_fit=0,new_AB_fit=0,adoption=False,submission_created=False,DIAG_EXPANSION_SCREEN=False,PASS2_ADOPTION_HOLD_FLAG=True,report='실험결과_v4.md',critic='최종혹독비평_v1.md',critic_correction='최종혹독비평_보고서대조_v2.md',report_links_checked=len(links),files={n:sha(H/n) for n in files},scope_notes=['public360days only; public pass2 is46days, actual evaluation60days untested','selected161 support600tree detailed trace onlyfold1seed7','root tree routing; independent critic arithmetic+source/receipt review, not duplicate routing','CI95 contains0; screen rejection is not a population-significance claim'])
with (H/'완료상태_v2.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print('FINAL_TWO_H0_COMPLETION_VERIFIED',len(rows),flush=True)

