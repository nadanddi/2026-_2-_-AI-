from pathlib import Path
import pandas as pd
import numpy as np
import json,math,hashlib,html

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
CASES=[('F13',194),('F47',191)]
LABELS={'out_temp':'외기 온도','out_hum':'외기 습도','out_rad':'외부 일사','out_wspd':'외부 풍속','in_temp':'실내 온도','in_hum':'실내 습도','in_co2':'실내 CO₂','in_rad':'실내 일사','act_vent':'환기 구동값','act_side':'측창 구동값','act_shade':'차광 구동값','act_thermal':'보온 구동값','act_valve':'밸브 구동값','act_heating':'난방 구동값','act_circfan':'순환팬 구동값','act_co2':'CO₂ 구동값','act_fog':'포그 구동값','act_cool':'냉방 구동값','act_pump':'펌프 구동값','sub_temp':'배지 온도 정답','sub_ec_public':'EC 공개 정답','temp_BASE':'온도 BASE 예측','temp_CODEX':'온도 CODEX 예측','temp_PFN':'온도 PFN 예측','temp_W30G':'온도 최종 예측','temp_error':'온도 최종 오차(예측−정답)','ec_season_v2_public':'EC 최종 공개검증 예측'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def val(v):return None if pd.isna(v) else float(v)

def main():
    source=ROOT/'공용/대회자료/정형데이터/참가자_배포'
    x=pd.read_csv(source/'train_X.csv');rawcols=[c for c in x if c!='row_id'];assert len(rawcols)==19
    ids=[f'{f}_{d:03d}_{h:02d}' for f,d in CASES for h in range(24)]
    a=x[x.row_id.isin(ids)].set_index('row_id').reindex(ids).reset_index();assert a.row_id.nunique()==48 and len(a)==48
    ty=pd.read_csv(source/'train_y.csv',usecols=['row_id','sub_temp']);a=a.merge(ty,on='row_id',validate='one_to_one')
    e=pd.read_csv(ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv');e=e[(e.validator=='DIAG10')&(e.seed==7)&e.row_id.isin(ids)][['row_id','sub_ec','season_v2']].rename(columns={'sub_ec':'sub_ec_public','season_v2':'ec_season_v2_public'})
    a=a.merge(e,on='row_id',how='left',validate='one_to_one')
    t=pd.read_csv(ROOT/'집/코덱스/local/temp_tk_season_20261003_v1/TK1_predictions.csv');t=t[(t.validator=='DIAG10')&(t.base_seed==7)&(t.context=='1-8')&t.row_id.isin(ids)&t.member.isin(['BASE','CODEX','PFN','W30G'])]
    assert t.groupby('row_id').size().eq(4).all()
    p=t.pivot(index='row_id',columns='member',values='prediction').add_prefix('temp_').reset_index();a=a.merge(p,on='row_id',validate='one_to_one')
    a['temp_error']=a.temp_W30G-a.sub_temp;a['farm']=a.row_id.str[:3];a['day']=a.row_id.str[4:7].astype(int);a['hour']=a.row_id.str[8:10].astype(int)
    fields=rawcols+['sub_temp','sub_ec_public','temp_BASE','temp_CODEX','temp_PFN','temp_W30G','temp_error','ec_season_v2_public']
    a=a[['row_id','farm','day','hour']+fields]
    a.to_csv(HERE/'두날_전체원자료와예측_v1.csv',index=False,encoding='utf-8-sig')
    pairs=[];summary=[];metadata={};panels=[]
    left=a[a.farm=='F13'].sort_values('hour').set_index('hour');right=a[a.farm=='F47'].sort_values('hour').set_index('hour')
    for col in fields:
        u=left[col];v=right[col];both=u.notna()&v.notna();eq=both&(u==v);missing=u.isna()&v.isna();only=u.isna()^v.isna();different=both&~eq
        status='모두 결측' if missing.all() else '한쪽 공개값 없음' if only.all() else '24시간 완전 동일' if eq.all() else '일부 동일·일부 다름' if eq.any() else '24시간 모두 다름'
        group='입력' if col in rawcols else '정답' if col in ['sub_temp','sub_ec_public'] else '예측·오차'
        s=dict(column=col,label=LABELS[col],group=group,status=status,same_hours=int(eq.sum()),different_hours=int(different.sum()),both_missing_hours=int(missing.sum()),one_missing_hours=int(only.sum()),F13_mean=val(u.mean()),F47_mean=val(v.mean()),max_abs_difference=val((v-u)[both].abs().max()))
        summary.append(s)
        values=[]
        for hour in range(24):
            row=dict(column=col,label=LABELS[col],group=group,hour=hour,F13_row_id=left.loc[hour,'row_id'],F47_row_id=right.loc[hour,'row_id'],F13=val(u.loc[hour]),F47=val(v.loc[hour]),difference_F47_minus_F13=val(v.loc[hour]-u.loc[hour]),comparison='양쪽 결측' if missing.loc[hour] else '공개값 없음' if only.loc[hour] else '같음' if eq.loc[hour] else '다름')
            pairs.append(row);values.append(row)
        panels.append(dict(**s,values=values))
    pd.DataFrame(pairs).to_csv(HERE/'시간별_항목별_대조_v1.csv',index=False,encoding='utf-8-sig');pd.DataFrame(summary).to_csv(HERE/'열별_동일여부_요약_v1.csv',index=False,encoding='utf-8-sig')
    for f,d in CASES:
        q=a[a.farm==f];error=q.temp_error.to_numpy()
        metadata[f]=dict(farm=f,day=d,hours=24,temperature_rmse=float(np.sqrt(np.mean(error**2))),temperature_truth_mean=float(q.sub_temp.mean()),temperature_prediction_mean=float(q.temp_W30G.mean()),air_mean=float(q.in_temp.mean()),humidity_mean=float(q.in_hum.mean()),ec_public_hours=int(q.sub_ec_public.notna().sum()))
    result=dict(cases=metadata,raw_input_columns=rawcols,all_fields=fields,panels=panels,source_hashes=dict(train_X=sha(source/'train_X.csv'),temperature_public_oof=sha(ROOT/'집/코덱스/local/temp_tk_season_20261003_v1/TK1_predictions.csv'),ec_public_oof=sha(ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv')),ec_note='F13 194일 EC는 기존 공개 OOF에 없어서 미열람. EC 잠금 파일/정답은 읽지 않았다.',temperature_model='DIAG10 W30G BASE7/CODEX726/PFN문맥1–8')
    (HERE/'data.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(pd.DataFrame(summary)[['column','status','same_hours','different_hours','F13_mean','F47_mean','max_abs_difference']].to_string(index=False))

if __name__=='__main__':main()
