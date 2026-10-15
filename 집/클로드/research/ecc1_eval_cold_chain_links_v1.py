import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd, numpy as np, os
EL = pd.read_csv(r"C:\work\farmai\제출\미제출\submission_29_27온도사슬M1_EC앵커확률비례_가설검정\팜모니_정형데이터_재현패키지_29\output_29\fl_eval_links.csv")
E = pd.read_csv(os.path.join(env.HERE if hasattr(env,'HERE') else '.', "emap1_eval_map_v1.csv")) if False else None
TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TE = TE[TE.row_id.str[:3].isin(["F13","F47"])]; TE["rec"] = TE.row_id.str[:7]
mn = TE.groupby("rec").in_temp.min()
V10 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "eval60_predecessor_v10.csv")); V15 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_eval.csv"))
print(V10.columns.tolist()[:12]); print(V15.columns.tolist()[:12])
EL["min_in"] = EL.rec.map(mn)
ch = EL[(~EL.prev_is_labelled.astype(bool)) & (EL.p0 >= .5) & (EL.dT_air.abs() <= 1.5)]
ch = ch[[isinstance(p, str) and p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) for r, p in zip(ch.rec, ch.prev0)]]
v10 = V10.set_index(V10.columns[0]); v15 = V15.set_index(V15.columns[0])
def get(df, r, cols):
    if r not in df.index: return None
    for c in cols:
        if c in df.columns: return df.at[r, c]
for _, z in ch.sort_values("min_in").iterrows():
    print(z.rec, "min %.1f" % z.min_in, "FL prev", z.prev0, "p %.2f" % z.p0, "dT %.1f" % z.dT_air, "| v10", get(v10, z.rec, ["prev", "pred", "prev0", "a"]), "| v15r", get(v15, z.rec, ["prev0", "prev", "a"]))
