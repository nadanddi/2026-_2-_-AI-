from pathlib import Path
import sys, csv, json, shutil, hashlib, zipfile
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
OUT=ROOT/'집/코덱스/local/ec_v2_base_delivery_20261002_v1/artifact'
OLD=ROOT/'집/코덱스/local/ec_v2_state_20261001_v1/artifact_numpy253'
STAGE=OUT/'stage'
REFERENCE=OLD/'reproduction_check/replay/prediction_details.npz'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def zip_dir(path,folder):
    assert not path.exists()
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(folder).as_posix())
def main():
    assert not OUT.exists(),'새 경로에만 생성합니다'
    assert sha(REFERENCE)=='36fd8bef6218b158cd1d75c06f67f107430d651d2d3101832cf5fe4bb661ccd5'
    (STAGE/'data').mkdir(parents=True)
    for name in ['train_X.csv','train_y.csv','test_X.csv','sample_submission.csv']:
        shutil.copyfile(OLD/'stage/data'/name,STAGE/'data'/name)
    for name in ['env.py','env_extra.py','requirements.txt','locked_days.json','tabpfn-v2-regressor.ckpt']:
        shutil.copyfile(OLD/'stage'/name,STAGE/name)
    shutil.copytree(OLD/'stage/dlls',STAGE/'dlls')
    shutil.copyfile(HERE/'ec_model_base_v1.py',STAGE/'ec_model_base_v1.py')
    shutil.copyfile(HERE/'PROTOCOL.md',STAGE/'PROTOCOL.md')
    with np.load(REFERENCE) as z:ids=z['row_id'].copy();pred=z['baseline'].copy()
    sample=pd.read_csv(STAGE/'data/sample_submission.csv')
    assert ids.tolist()==sample.row_id.tolist() and len(ids)==1440 and np.isfinite(pred).all()
    answer=pd.DataFrame({'row_id':ids,'sub_ec':pred})
    answer.to_csv(OUT/'ec_v2_base_v1.csv',index=False,float_format='%.17g',encoding='utf-8-sig')
    shutil.copyfile(OUT/'ec_v2_base_v1.csv',STAGE/'ec_v2_base_v1.csv')
    with (STAGE/'data/train_y.csv').open(encoding='utf-8-sig',newline='') as f:labels=list(csv.DictReader(f))
    locks={(z['farm'],int(z['day'])) for z in json.loads((STAGE/'locked_days.json').read_text(encoding='utf-8'))['selected']}
    assert len(labels)==8640 and all((r['row_id'].split('_')[0],int(r['row_id'].split('_')[1])) not in locks for r in labels)
    assert list(labels[0])==['row_id','sub_ec']
    wrapper=(OLD/'stage/reproduce.py').read_text(encoding='utf-8').replace("HERE/'ec_model.py'","HERE/'ec_model_base_v1.py'")
    (STAGE/'reproduce.py').write_text(wrapper,encoding='utf-8')
    shutil.copyfile(OLD/'stage/merge_ec.py',STAGE/'merge_ec.py')
    provenance={'status':'PREPARED_NOT_REPRODUCTION_VERIFIED','reference_cache':str(REFERENCE),
                'reference_cache_sha256':sha(REFERENCE),'source_model_sha256':sha(OLD/'stage/ec_model.py'),
                'code_sha256':sha(HERE/'ec_model_base_v1.py'),'protocol_sha256':sha(HERE/'PROTOCOL.md'),
                'csv_sha256':sha(OUT/'ec_v2_base_v1.csv'),'rows':1440,'training_rows':7344,'training_days':306,
                'state_correction':0,'temperature_created':False,'platform_submission':False,'locked_labels_included':False,
                'files_sha256':{p.relative_to(STAGE).as_posix():sha(p) for p in STAGE.rglob('*') if p.is_file()}}
    save(OUT/'preparation_manifest.json',provenance)
    zip_dir(OUT/'clean_check_input.zip',STAGE)
    extract=OUT/'clean_check_extract'
    with zipfile.ZipFile(OUT/'clean_check_input.zip') as z:z.extractall(extract)
    assert all(sha(extract/p)==h for p,h in provenance['files_sha256'].items())
    print(json.dumps({'stage':str(STAGE),'extract':str(extract),'rows':1440,'csv_sha256':provenance['csv_sha256']},ensure_ascii=False))
if __name__=='__main__':main()
