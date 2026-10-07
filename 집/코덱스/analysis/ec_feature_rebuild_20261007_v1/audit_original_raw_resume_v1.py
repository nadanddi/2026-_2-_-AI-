"""Synthetic resume contracts only: never fit, never read heldout labels."""
from pathlib import Path
import importlib.util,tempfile,json,hashlib,copy
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prospective_original_raw',HERE/'run_domain_original_raw_v3.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
tests=[]
with tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic_raw_resume_') as temp:
    root=Path(temp);r.HERE=root
    prep=root/'checkpoints/DOMAIN24_ORIGINAL_PREPARATION_v2';prep.mkdir(parents=True)
    (prep/'complete.json').write_text('synthetic only',encoding='utf-8')
    regpath=root/'DOMAIN24_original_raw_fit_registration_v3.json';regpath.write_text('synthetic registration only',encoding='utf-8')
    loader=str(Path(r.features_module.__file__).resolve())
    model=r.model_module.model('ET',47)
    params=r.model_contract(model)
    # Enforce that any unexpected non-resume route cannot actually fit.
    def fail_fit(*args,**kwargs):raise RuntimeError('Synthetic audit prohibits model fitting')
    model.fit=fail_fit
    r.model_module.model=lambda name,seed:model
    reg={'source_sha256':{loader:sha(loader)},'runtime_module_paths':r.runtime_paths(),
         'preparation_complete_sha256':sha(prep/'complete.json'),'model_contracts':{'ET_seed47':params}}
    tx=r.pd.DataFrame({'a':[1.,3.,5.],'empty':[float('nan')]*3})
    qx=r.pd.DataFrame({'a':[2.,4.],'empty':[float('nan')]*2})
    y=r.np.asarray([.1,.2,.3]);ids=['SYNTHETIC_A','SYNTHETIC_B'];probes={0:{'BASELINE_ET':qx.iloc[[0]]}}
    details={'require_all66':True,'loader_source_sha256':sha(loader),
             'train_label_sha256':hashlib.sha256(y.astype('<f8').tobytes()).hexdigest(),
             'preparation_receipt_sha256':'synthetic_receipt'}
    signature={'train_matrix_sha256':r.features_module.matrix_sha(tx),
               'query_matrix_sha256':r.features_module.matrix_sha(qx),
               'train_labels_sha256':details['train_label_sha256'],'columns':list(tx.columns)}
    contract={'registration_sha256':sha(regpath),'validator':'SYNTHETIC','fold':0,'seed':47,
              'member':'ET','candidate':'BASELINE_ET','preparation_receipt_sha256':'synthetic_receipt',
              'matrices':signature,'model_contract':params,'runtime_module_paths':r.runtime_paths()}
    stats=r.canonical_numeric_sha(r.np.asarray([3.,float('nan')]))
    record={'status':'ORIGINAL_RAW_MODEL_PREDICTION_PASS_NO_SCORE','contract':contract,'row_ids':ids,
            'pred':[.2,.3],'pred_sha256':r.digest([.2,.3]),'heldout_truth_loaded':False,
            'imputer':{'statistics_sha256':stats,'independent_train_median_sha256':stats,
                       'all_missing_train_columns':['empty'],'effective_columns':['a'],'fit_reference_only':True},
            'audit':{'prefix_prediction_checks':1,'all_errors_max':0.,
                     'reverse_scattered_prefix_max_differences':[0.,0.,0.],'fit_rows':3,'query_rows':2}}
    path=root/'BASELINE_ET_seed47.json'
    def invoke(payload,labels=y,ctx=details):
        path.write_text(json.dumps(payload),encoding='utf-8')
        return r.fit_one(reg,root,'SYNTHETIC',0,47,'ET','BASELINE_ET',tx,qx,labels,ids,probes,ctx)
    assert invoke(record)==path;tests.append({'case':'valid_resume_no_fit','PASS':True})
    variants={}
    bad=copy.deepcopy(record);bad['imputer']['statistics_sha256']='0'*64;variants['imputer_hash']=bad
    bad=copy.deepcopy(record);bad['imputer']['effective_columns']=['a','empty'];variants['effective_columns']=bad
    bad=copy.deepcopy(record);bad['imputer']['fit_reference_only']=False;variants['reference_only_flag']=bad
    bad=copy.deepcopy(record);bad['audit']['fit_rows']=2;variants['fit_rows']=bad
    bad=copy.deepcopy(record);bad['audit']['reverse_scattered_prefix_max_differences']=[0.,0.];variants['missing_prefix_error']=bad
    bad=copy.deepcopy(record);bad['audit']['reverse_scattered_prefix_max_differences'][0]=-1.;variants['negative_error']=bad
    bad=copy.deepcopy(record);bad['audit']['reverse_scattered_prefix_max_differences'][0]=float('nan');variants['nonfinite_error']=bad
    bad=copy.deepcopy(record);bad['audit']['all_errors_max']=1e-7;variants['inconsistent_max']=bad
    bad=copy.deepcopy(record);bad['contract']['model_contract'][0]['parameters']['strategy']='mean';variants['model_contract']=bad
    bad=copy.deepcopy(record);bad['heldout_truth_loaded']=True;variants['heldout_flag']=bad
    for name,bad in variants.items():
        try:invoke(bad)
        except AssertionError:tests.append({'case':name+'_rejected','PASS':True})
        else:raise AssertionError(name+' accepted')
    for name,labels,ctx in [('actual_y_mismatch',y+.01,details),
                            ('probe_bypass',y,{**details,'require_all66':False})]:
        try:invoke(record,labels,ctx)
        except AssertionError:tests.append({'case':name+'_rejected','PASS':True})
        else:raise AssertionError(name+' accepted')
result={'status':'SYNTHETIC_ORIGINAL_RAW_RESUME_PASS','code_sha256':sha(__file__),
        'runner_sha256':sha(HERE/'run_domain_original_raw_v3.py'),'tests':tests,
        'model_fits':0,'real_query_predictions':0,'real_heldout_labels_read':0,
        'limits':['Synthetic contract checks, not actual66 preparation completion or full model gate']}
with (HERE/'DOMAIN24_original_raw_resume_synthetic_audit_v1.json').open('x',encoding='utf-8') as h:
    json.dump(result,h,ensure_ascii=False,indent=2)
print(f'Synthetic original resume {len(tests)} PASS; fitting prohibited, no real query or labels')
