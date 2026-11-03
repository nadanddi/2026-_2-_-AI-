"""Inspect original ABL3 plans in memory; deny writes outside this review directory."""
import sys,os,builtins,contextlib,json,ast
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
OWN=Path(__file__).resolve().parent
SRC=ROOT/'연구실/클로드/code/abl3_three_conditions_v1.py'
raw=SRC.read_text(encoding='utf-8')
realopen=builtins.open
def safeopen(file,mode='r',*args,**kw):
 if isinstance(file,(str,bytes,os.PathLike)) and any(c in mode for c in 'wax+'):
  q=Path(file).resolve()
  if not q.is_relative_to(OWN):raise PermissionError('READONLY REVIEW blocked write: '+str(q))
 return realopen(file,mode,*args,**kw)
builtins.open=safeopen
with safeopen(OWN/'prefix_probe_v1.log','x',encoding='utf-8') as log,contextlib.redirect_stdout(log):
 g={'__file__':str(SRC),'__name__':'q03_readonly_probe'}
 exec(compile(raw[:raw.index('TROWS = []; TBLK = {}')],str(SRC),'exec'),g)
 summary={}
 for L,F in g['FR'].items():
  F['L']=L;summary[L]={}
  for gate in [True,False]:
   A,C,order,stat=g['oracle_temp'](F,gate)
   color={};cycles=[]
   def dfs(r,stack):
    if color.get(r)==1:
     cycles.append(stack[stack.index(r):]+[r]);return
    if color.get(r)==2:return
    color[r]=1
    if r in C:dfs(C[r],stack+[r])
    color[r]=2
   for r in F['pos']:dfs(r,[])
   ix={r:i for i,r in enumerate(order)}
   bad=[(r,p) for r,p in C.items() if ix[p]>=ix[r]]
   cross=[(r,p) for r,p in C.items() if r[:3]!=p[:3]]
   FO=dict(zip(F['key'],F['M'].fold.values))
   crossfold=[(r,p) for r,p in C.items() if FO[r]!=FO[p]]
   summary[L]['O1' if gate else 'O2']={'anchors':len(A),'chains':len(C),'cycles':cycles,'predecessor_order_violations':bad,'cross_id_chains':cross,'cross_fold_chains':crossfold,'stat':stat}
 print(json.dumps(summary,ensure_ascii=False,indent=2))
with safeopen(OWN/'plan_probe_v1.json','x',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False))
