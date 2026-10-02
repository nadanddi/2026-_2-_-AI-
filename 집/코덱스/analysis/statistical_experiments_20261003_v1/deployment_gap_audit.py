from support import *
def main():
    sx=pd.read_csv(Path(env.DATA)/'test_X.csv',usecols=['row_id']);sx=sx[sx.row_id.str[:3].isin(['F13','F47'])].copy();sx['farm']=sx.row_id.str[:3];sx['day']=sx.row_id.str[4:7].astype(int)
    lab,_,_,_,_=loadec();rawids=pd.read_csv(Path(env.DATA)/'train_X.csv',usecols=['row_id']);allkeys={(r[:3],int(r[4:7])) for r in rawids.row_id if r[:3] in ['F13','F47']};publickeys=set(lab[['farm','day']].itertuples(index=False,name=None));excluded=allkeys-publickeys;assert len(excluded)==40
    banned={(f,int(d)+j) for f,d in excluded for j in [-1,0,1]};keep=np.array([(f,int(d)) not in banned for f,d in zip(lab.farm,lab.day)]);tr=lab[keep];assert len(tr)==7344 and len(tr[['farm','day']].drop_duplicates())==306
    records=dict(public360=gapstats(lab,sx),base_fit306=gapstats(tr,sx));savej(HERE/'deployment_gap_audit.json',dict(status='METADATA_ONLY',records=records,source='public OOF IDs; exclusions inferred from input IDs, no lock file or EC lock targets read',code_hash=sha(__file__)));print(json.dumps(records),flush=True)
if __name__=='__main__':main()
