import os, hashlib, numpy as np, pandas as pd
P = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22"
D = r"C:\work\farmai\공용\대회자료\정형데이터\참가자_배포"
sub = pd.read_csv(P + r"\submission_22.csv"); t22 = pd.read_csv(P + r"\온도\temp_candidate_v22.csv")
m = sub.merge(t22, on="row_id", suffixes=("", "_c"))
print("[4] submission_22 sub_temp == temp_candidate_v22 (6dp):", np.allclose(m.sub_temp, m.sub_temp_c, atol=5e-7), "rows", len(m))
u = r"C:\work\farmai\제출\미제출\submission_22_측정감쇠앵커_v7EC앵커_가설검정"
for f in os.listdir(u):
    if f.endswith(".csv"):
        x = pd.read_csv(os.path.join(u, f));
        if "sub_temp" in x: print("   미제출 folder", f, "temp equal:", np.allclose(x.set_index("row_id").loc[sub.row_id].sub_temp.values, sub.sub_temp.values, atol=5e-7))
v13 = pd.read_csv(P + r"\온도\temp_candidate_v13.csv"); v18 = pd.read_csv(P + r"\온도\temp_candidate_v18.csv")
npz = np.load(P + r"\온도\output\temp_candidate_v17_members.npz", allow_pickle=True); print("   v17 npz keys", list(npz.keys()))
tX = pd.read_csv(os.path.join(D, "train_X.csv")); ty = pd.read_csv(os.path.join(D, "train_y.csv")); sX = pd.read_csv(os.path.join(D, "test_X.csv"))
for d in (tX, sX):
    d["farm"] = d.row_id.str[:3]; d["day"] = d.row_id.str[4:7].astype(int); d["hour"] = d.row_id.str[8:10].astype(int)
tr = tX[tX.farm.isin(["F13", "F47"])].merge(ty[["row_id", "sub_temp"]], on="row_id")
tr = tr[tr.in_temp.notna() & tr.sub_temp.notna()]
E = sX[["row_id", "farm", "day", "hour", "in_temp"]].merge(sub[["row_id", "sub_temp"]], on="row_id")
E = E.merge(v13.rename(columns={"sub_temp": "p13"})[["row_id", "p13"]], on="row_id").merge(v18.rename(columns={"sub_temp": "p18"})[["row_id", "p18"]], on="row_id")
if "base" in npz:
    mm = pd.DataFrame({"row_id": npz["row_id"], "base": npz["base"], "codex": npz["codex"], "pfn": npz["pfn_samples"].mean(0) if "pfn_samples" in npz else np.nan})
    E = E.merge(mm, on="row_id", how="left")
te_days = set(zip(sX.farm, sX.day))
E["start"] = [(f, d - 1) not in te_days for f, d in zip(E.farm, E.day)]
# matched training expectation: for each eval row, mean(sub_temp - in_temp) of training rows same farm, same hour, in_temp within +-1C
tr["gap"] = tr.sub_temp - tr.in_temp
exp = []
G = {k: g for k, g in tr.groupby(["farm", "hour"])}
for f, h, t in zip(E.farm, E.hour, E.in_temp):
    g = G[(f, h)]; s = g.gap[(g.in_temp - t).abs() <= 1.0]
    if len(s) < 15: s = g.gap[(g.in_temp - t).abs() <= 2.0]
    exp.append((s.mean(), len(s)))
E["exp_gap"] = [e[0] for e in exp]; E["n_match"] = [e[1] for e in exp]
E["gap"] = E.sub_temp - E.in_temp; E["dev"] = E.gap - E.exp_gap
print("\n[5] eval in_temp mean %.2f (train %.2f); eval rows with <15 matched train rows within 1C: %d" % (E.in_temp.mean(), tr.in_temp.mean(), int((E.n_match < 15).sum())))
pd.set_option("display.width", 250)
T = E.groupby(["farm", "hour"]).agg(pred_gap=("gap", "mean"), train_matched_gap=("exp_gap", "mean"), dev=("dev", "mean"), in_temp=("in_temp", "mean")).round(2)
print(T.unstack(0).to_string())
print("\nby farm: mean dev", E.groupby("farm").dev.mean().round(3).to_dict(), " |dev| > 1C rows:", int((E.dev.abs() > 1).sum()))
print("\nchunk-start records vs others, hours 0-7 (dev = pred gap - matched train gap):")
x = E[E.hour <= 7].groupby(["start", "hour"]).dev.mean().unstack(0).round(3); print(x.to_string())
print("rows-level: start h0-4 mean dev %.3f ; non-start h0-4 %.3f" % (E[(E.start) & (E.hour <= 4)].dev.mean(), E[(~E.start) & (E.hour <= 4)].dev.mean()))
if "base" in E:
    for c in ("base", "codex", "pfn", "p13"):
        E["d_" + c] = E[c] - E.in_temp - E.exp_gap
    print("\nmember dev (member - in_temp - matched train gap), start vs non-start, hours 0-4:")
    print(E[E.hour <= 4].groupby("start")[["d_base", "d_codex", "d_pfn", "d_p13"]].mean().round(3).to_string())
    print("hours 5-23:"); print(E[E.hour > 4].groupby("start")[["d_base", "d_codex", "d_pfn", "d_p13"]].mean().round(3).to_string())
E.to_csv(r"C:\work\farmai\집\클로드\research\audit_temp\eval_sanity.csv", index=False)
# per record dev table
R = E.groupby(["farm", "day"]).agg(dev=("dev", "mean"), in_temp=("in_temp", "mean"), start=("start", "first")).round(2)
print("\nper-record mean dev (top |dev|):"); print(R.reindex(R.dev.abs().sort_values(ascending=False).index).head(12).to_string())
