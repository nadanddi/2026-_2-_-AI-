"""Inference for the saved experimental high-day classifier; current-prefix inputs."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
import recipe_v1 as C
import numpy as np,pandas as pd,joblib

def predict(model_dir,x):
 model_dir=Path(model_dir);m=json.loads((model_dir/'model_manifest_v1.json').read_text(encoding='utf8'));assert m['feature_columns']==C.COLS and m['ec_threshold']==C.EC_THRESHOLD
 f=C.features(x[['row_id']+C.RAW]);scores=[]
 for rec in m['models']:
  model=joblib.load(model_dir/rec['file']);scores.append(C.positive_score(model,f[C.COLS]))
 p=np.mean(scores,axis=0);assert np.isfinite(p).all()
 return pd.DataFrame({'row_id':f.row_id,'high_ec_score':p,'high_ec_class':(p>=.5).astype(int)})
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);a=parser.parse_args();out=Path(a.output);assert not out.exists(),str(out)
 x=pd.read_csv(a.input);result=predict(a.model,x);out.parent.mkdir(parents=True,exist_ok=True);result.to_csv(out,index=False)
