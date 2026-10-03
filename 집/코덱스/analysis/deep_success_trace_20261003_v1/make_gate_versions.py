from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'collect_pfn_v2.py').read_text(encoding='utf-8')
s=s.replace("cp=(.4+.1*g)*p['BASE']+(.6-.4*g)*p['CODEX']+.3*g*pf;allp=cp+.3*g*agg[key]", "cg=S.gate(change);cp=(.4+.1*cg)*p['BASE']+(.6-.4*cg)*p['CODEX']+.3*cg*pf;allp=cp+.3*cg*agg[key]")
for name in ['whole_model_effects','whole_model_effect_hours','whole_model_effect_summary','pfn_context_effects']:s=s.replace(name+'.csv',name+'_v2.csv')
s=s.replace('pfn_supplement_verification.json','pfn_supplement_verification_v2.json')
(H/'collect_pfn_v3.py').write_text(s,encoding='utf-8')
v=(H/'verify_whole.py').read_text(encoding='utf-8')
for name in ['whole_model_effects','whole_model_effect_hours']:v=v.replace(name+'.csv',name+'_v2.csv')
v=v.replace('whole_verification.json','whole_verification_v2.json')
(H/'verify_whole_v2.py').write_text(v,encoding='utf-8')
