import pandas as pd, numpy as np
P = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\온도"
a = pd.read_csv(P + r"\temp_candidate_v18.csv"); b = pd.read_csv(P + r"\temp_candidate_v22.csv")
A = pd.read_csv(r"C:\work\farmai\집\클로드\research\local\linker_v7\eval60_confident_1step_anchor_v7.csv")
d = b.sub_temp - a.sub_temp; k = a.row_id.str[:7]
s = pd.DataFrame({"k": k, "h": a.row_id.str[8:10].astype(int), "d": d})
S = s.groupby("k").agg(d0=("d", "first"), dmean=("d", "mean")).join(A.set_index("row")[["prev_h0", "p_h0", "dT_air", "anchor_st23"]], how="inner")
S["prev_gap_days"] = [int(r[4:7]) - int(p[4:7]) if r[:3] == p[:3] else np.nan for r, p in zip(S.index, S.prev_h0)]
print(S.round(3).to_string())
print("anchored records:", int((S.d0.abs() > 1e-9).sum()), "| with predecessor across a deleted record (gap 2):", S[S.prev_gap_days == 2].index.tolist())
