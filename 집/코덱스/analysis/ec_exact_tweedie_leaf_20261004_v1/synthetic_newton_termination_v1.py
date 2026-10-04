"""Regression fixture reproducing stationary Newton bracket jump; no real data."""
from pathlib import Path
import sys,math,json,ast,hashlib
H=Path(__file__).resolve().parent;ROOT=H.parents[3];sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
from whole_cpp_recheck_v2 import safeguarded_newton
tree=ast.parse((H/'diagnose_newton_decimal_v1.py').read_text(encoding='utf-8'))
ns=dict(math=math);node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='old']
exec(compile(ast.Module(body=node,type_ignores=[]),'old_solver_only','exec'),ns)
A=48.61579194990402;B=49.83992556463348;expected=-.024372825077840032119143319293672123365638388402882945403698517910019228220125714
old,trail=ns['old'](A,B);new=safeguarded_newton(A,B)
assert abs(old-expected)>1e-12 and abs(new-expected)<=1e-12
neg=A*math.exp(-new/2);pos=B*math.exp(new/2)
assert abs(-neg+pos+new)<=1e-12*max(1,neg+pos+abs(new))
assert 2*neg+2*pos+.5*new*new<=2*A+2*B+1e-12*max(1,2*A+2*B)
cpp=json.loads((H/'synthetic_cpp_whole_v2.json').read_text(encoding='utf-8'));assert len(cpp['rejected'])==8
r=dict(status='PASS_TERMINATION_REGRESSION_AND_8_CORRUPTIONS',A=A,B=B,expected_decimal80_root=expected,old100=old,old_gap=abs(old-expected),new=new,new_gap=abs(new-expected),old_first10=trail[:10],old_last15=trail[-15:],corruptions=cpp['rejected'],absolute_root_atol=1e-12,residual_relative_atol=1e-12,helper_sha256=hashlib.sha256((H/'whole_cpp_recheck_v2.py').read_bytes()).hexdigest(),actual_fit=0,actual_predict=0,outer_score=0,raw_ec_reads=0,test_reads=0,EL1_reads=0)
with (H/'synthetic_newton_termination_result_v1.json').open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
print(r['status'],r['old_gap'],r['new_gap'])
