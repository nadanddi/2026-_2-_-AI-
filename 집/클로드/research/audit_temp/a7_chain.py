import sys, os, numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\온도\code")
import make_temp_v18_chain as V
D = r"C:\work\farmai\공용\대회자료\정형데이터\참가자_배포"
tX = V.ident(pd.read_csv(os.path.join(D, "train_X.csv"))); sX = V.ident(pd.read_csv(os.path.join(D, "test_X.csv")))
records = sorted(set(zip(tX.farm, tX.day)) | set(zip(sX.farm, sX.day)))
sd = tX[V.V].std()
CM = V.chain_table(tX, records, sd); CF = V.chain_table(pd.concat([tX, sX]), records, sd)
ev = set(zip(sX.farm, sX.day)); trs = set(zip(tX.farm, tX.day))
CF = CF[[ (f, d) in ev for f, d in zip(CF.farm, CF.day)]]
CM = CM[[ (f, d) in trs for f, d in zip(CM.farm, CM.day)]]
# literal previous existing record distance
def litgap(df):
    out = []
    for f, d in zip(df.farm, df.day):
        days = [dd for ff, dd in records if ff == f and dd < d]
        out.append(d - days[-1] if days else np.nan)
    return np.array(out)
CF["gap"] = litgap(CF); CM["gap"] = litgap(CM)
print("eval records: lit1 gap counts", pd.Series(CF.gap).value_counts().to_dict(), "| lit1 NaN", int(CF.lit1_in23.isna().sum()))
print("train records (MASK): lit1 gap counts", pd.Series(CM.gap).value_counts().to_dict(), "| lit1 NaN", int(CM.lit1_in23.isna().sum()), "| lit1 present with gap 2:", int(((CM.gap == 2) & CM.lit1_in23.notna()).sum()))
s = CF[CF.gap == 2]
print(s[["farm", "day", "lit1_in23", "ch_in23", "ch_cost"]].round(2).to_string(index=False))
print("train ch_cost median %.2f, eval chunk-start ch_cost median %.2f, eval other %.2f" % (CM.ch_cost.median(), s.ch_cost.median(), CF[CF.gap == 1].ch_cost.median()))
