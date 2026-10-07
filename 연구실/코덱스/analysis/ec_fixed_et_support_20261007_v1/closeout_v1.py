from pathlib import Path
import json,hashlib,re
H=Path(__file__).resolve().parent;ROOT=H.parents[3];L=ROOT/'연구실/코덱스/local'/H.name
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    c=json.loads((H/'completion_v1.json').read_text(encoding='utf-8'));assert c['new_ET_fits']==5 and c['other_fits']==0
    for name,digest in c['files'].items():assert sha(H/name)==digest
    for name in ['audit_tree_v1.json','audit_endpoint_v2.json','audit_prefix_v1.json','critic_arithmetic_v2.json']:
        d=json.loads((H/name).read_text(encoding='utf-8'));assert d['status'].startswith('PASS')
    assert (H/'최종보고서_비평대조_v1.md').exists()
    fits=0;locals={}
    for p in L.glob('*.npz'):
        m=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));assert m['sha']==sha(p);fits+=m.get('new_fit',0);locals[p.name]=m['sha']
    assert len(locals)==26 and fits==5
    text=(H/'비교결과_v3.md').read_text(encoding='utf-8');links=re.findall(r'\]\(([^)]+)\)',text)
    for link in links:assert (H/link).exists(),link
    required=['비교결과_v3.md','최종혹독비평_v1.md','최종보고서_비평대조_v1.md','보완기록_v1.md','completion_v1.json','audit_tree_v1.json','audit_endpoint_v2.json','audit_prefix_v1.json','critic_arithmetic_v2.json']
    out=dict(status='COMPLETE_FIXED_BASELINE_INPUT_SUPPORT_COMPARISON_REVIEWED',new_ET_fits=5,other_fits=0,adoption=False,submission=False,original_models=3,common_models=3,pair_comparisons=20,coalitions=18432,endpoint_checks=40,first_divergence_records=24000,recorded_shared_prefix_checks=23670,new_seed7_trees=1800,new_seed7_nodes=17004160,new_seed7_all_coalition_query_routing_rows=184320,raw_train_y_new_reads=0,report_links_checked=len(links),analysis_files={name:sha(H/name) for name in required},local_files=locals,local_bytes=sum(p.stat().st_size for p in L.iterdir() if p.is_file()),prereg_commit='5d9c0d5')
    with (H/'완료상태_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(out['status'],len(links),out['local_bytes'])
if __name__=='__main__':main()
