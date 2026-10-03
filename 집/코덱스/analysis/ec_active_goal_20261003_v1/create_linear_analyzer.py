from pathlib import Path
H=Path(__file__).resolve().parent;s=(H/'analyze.py').read_text(encoding='utf-8')
s=s.replace('r.baseline+.48*(shrink-r.old_et_shrunk)','.8*r.baseline+.2*shrink')
s=s.replace("(EH/'run.py')","(EH.parent/'run.py')")
p=H/'analyze_linear.py';assert not p.exists();compile(s,str(p),'exec');p.write_text(s,encoding='utf-8')
print(p)
