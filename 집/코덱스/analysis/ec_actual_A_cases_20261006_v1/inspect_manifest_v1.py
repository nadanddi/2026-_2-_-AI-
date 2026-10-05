import run_v2 as R
R.F.bootstrap(support=True)
refs,p,cols=R.F.preflight()
old=R.load(R.H.parent/'ec_final_output_loss_20261004_v1/preparation_v3.json')
def diff(a,b,path=''):
 if type(a)!=type(b): print(path,'TYPE',type(a).__name__,type(b).__name__);return
 if isinstance(a,dict):
  for k in a:diff(a[k],b[k],path+'/'+k)
 elif isinstance(a,list):
  for i,(x,y) in enumerate(zip(a,b)):diff(x,y,path+'/'+str(i))
 elif a!=b: print(path,repr(a),repr(b))
diff(p['manifest'],old['manifest'])
