"""Step 4: label-availability audit, without evaluating any new model."""
from support import *
def main():
    sx=pd.read_csv(Path(env.DATA)/'test_X.csv',usecols=['row_id']);sx=sx[sx.row_id.str[:3].isin(['F13','F47'])].copy();sx['farm']=sx.row_id.str[:3];sx['day']=sx.row_id.str[4:7].astype(int)
    records=[];lab,_,_,_,folds,_=loadtemp()
    for name,k,fd in folds:
        tm,vm=common.split_mask(lab,fd);tr,va=lab[tm],lab[vm];im,iv=inner(tr);records.append(dict(target='TEMP',validator=name,fold=k,scope='outer',**gapstats(tr,va)));records.append(dict(target='TEMP',validator=name,fold=k,scope='inner',**gapstats(tr[im],tr[iv])))
    records.append(dict(target='TEMP',validator='actual_metadata',fold=-1,scope='no_target_read',**gapstats(lab,sx)))
    lab,_,_,folds,_=loadec()
    for name,k,tm,vm in folds:
        tr,va=lab[tm],lab[vm];im,iv=inner(tr);records.append(dict(target='EC',validator=name,fold=k,scope='outer',**gapstats(tr,va)));records.append(dict(target='EC',validator=name,fold=k,scope='inner',**gapstats(tr[im],tr[iv])))
    records.append(dict(target='EC',validator='actual_metadata',fold=-1,scope='public_labels_only',**gapstats(lab,sx)))
    pd.DataFrame(records).to_csv(HERE/'gap_audit.csv',index=False);savej(HERE/'gap_audit.json',dict(status='AUDIT_ONLY',records=records,test_columns_read=['row_id'],test_values_read=False,split_models_retrained=False,code_hash=sha(__file__)))
    for r in records:
        if r['validator']=='actual_metadata':print(json.dumps(r),flush=True)
if __name__=='__main__':main()
