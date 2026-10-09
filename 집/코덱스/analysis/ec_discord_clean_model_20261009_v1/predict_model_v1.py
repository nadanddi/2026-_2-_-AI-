"""Predict with the frozen cleaned-data R3 model; no evaluation-time fitting."""
import sys,json,hashlib,argparse
from pathlib import Path
sys.dont_write_bytecode=True
import recipe_ec_v1 as C
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def predict_bundle(model_dir,raw):
    D=Path(model_dir);m=json.loads((D/'model_manifest_v1.json').read_text(encoding='utf8'))
    assert m['adoption'] is False and raw.row_id.is_unique
    for name,h in m['code_hashes'].items():assert sha(Path(__file__).parent/name)==h,name
    q=C.frozen_season(C.features(raw[['row_id']+C.RAW]),m['season_notes'],m['training_days'])
    raw_pred=np.zeros(len(q))
    for item in m['models']:
        p=D/item['file'];assert sha(p)==item['sha']
        model=joblib.load(p)
        with threadpool_limits(limits=2):z=np.asarray(model.predict(q[item['columns']]),float)
        raw_pred+=C.WEIGHTS[item['kind']]*z/len(C.SEEDS)
        del model
    pred=np.clip(C.shrink(raw_pred,q),*m['bounds']);assert np.isfinite(pred).all()
    return pd.DataFrame({'row_id':q.row_id,'prediction':pred})
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--model',type=Path,required=True);a.add_argument('--input',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    assert not args.output.exists();out=predict_bundle(args.model,pd.read_csv(args.input));out.to_csv(args.output,index=False)
