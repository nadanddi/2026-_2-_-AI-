from pathlib import Path
import json,hashlib,re,csv,datetime
H=Path(__file__).resolve().parent;ROOT=H.parents[3];L=ROOT/'연구실/코덱스/local'/H.name
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(n):return json.loads((H/n).read_text(encoding='utf-8-sig'))
base=read('baseline_receipt_v4.json');ab=read('ablation_receipt_v1.json');support=read('support_audit_v1.json');critic=read('critic_verify_ablation_v1.json');summary=read('critic_verify_diagnostic_summary_v1.json');prep=read('preparation_v4.json');ap=read('ablation_preparation_v1.json')
assert not (L/'baseline.lock').exists() and not (L/'ablation.lock').exists()
assert base['rows_sha']==sha(L/'baseline_rows.csv') and ab['rows_sha']==sha(L/'ablation_rows.csv')
assert prep['script_sha']==sha(H/'runner_v4.py') and prep['adapter_sha']==sha(H/'sg2_ref_v2.py') and prep['plan_sha']==sha(H/'PLAN_v2.md')
assert ap['source_sha']==ab['source_sha']==sha(H/'ablation_v1.py') and ab['preparation_sha']==sha(H/'ablation_preparation_v1.json')
assert critic['status']=='PASS_ABLATION_CACHE_PIPELINE_FSUM_SCORES_AND_FIXED_TAGS' and critic['rows']==69120 and critic['caches']==60
assert support['status']=='PASS_ALL_SAVED_TREE_PATHS_AND_SUPPORT' and support['total_nodes']==20321234
assert summary['status'].startswith('PASS')
assert ab['new_ET_fits']==57 and ab['additional_baseline_trace_fit']==1 and sum(m['new_fit'] for m in ab['models'])==57
counts={}
for name,expected in [('baseline_rows.csv',34560),('ablation_rows.csv',69120)]:
    with (L/name).open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    keys={(r['arm'],r['seed'],r['row_id']) for r in rows};assert len(rows)==len(keys)==expected
    counts[name]=dict(rows=len(rows),unique_row_ids=len({r['row_id'] for r in rows}),unique_days=len({(r['farm'],r['day']) for r in rows}))
    assert counts[name]['unique_row_ids']==8640 and counts[name]['unique_days']==360
report=H/'진단결과_v3.md';text=report.read_text(encoding='utf-8-sig');links=re.findall(r'\]\((C:/work/[^)]+)\)',text)
assert len(links)==7
for link in links:assert Path(link).exists(),link
assert '진행 중' not in text and '별도CSV 참조' not in text
assert (H/'최종혹독비평_v1.md').exists()
files=['진단결과_v3.md','최종혹독비평_v1.md','보완기록_v2.md','진단비교_v2.png','baseline_receipt_v4.json','ablation_receipt_v1.json','support_audit_v1.json','critic_verify_baseline_v1.json','critic_verify_ablation_v1.json','critic_verify_diagnostic_summary_v1.json','diagnostic_summary_v1.json','ablation_score_v1/completion.json']
out=dict(status='COMPLETE_CURRENT14_AND_ET_DELETION_DIAGNOSTICS',completed_at=datetime.datetime.now().isoformat(),counts=counts,new_fits=dict(baseline_classic=90,deletion_ET=57,baseline_trace_ET=1,PFN=0),source_registration=['b8d40fb','f570b01','42dd5ad'],selected_case_tags=read('ablation_score_v1/completion.json')['tags'],adoption=False,submission_created=False,new_followup_fit=0,final_report='진단결과_v3.md',independent_critic='최종혹독비평_v1.md',checked_report_links=len(links),files={n:sha(H/n) for n in files},limitations=['public360days only; actual evaluation60days/A/B/EL1/lock efficacy untested','tree numeric routing by root; independent critic reviews code/receipt and independently recomputes support summaries','selected161 diagnostic is not independent generalization evidence','physical EC cause and label defects unproven'])
with (H/'완료상태_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print('FINAL_COMPLETION_VERIFIED',counts,flush=True)
