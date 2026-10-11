# -*- coding: utf-8 -*-
"""EC audit 3: SG2 instrumentation, EA2 anchors, CSV equality, sanity of predicted levels."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
ROOT = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22"
PK = os.path.join(ROOT, "EC")
sys.path.insert(0, PK)
import numpy as np, pandas as pd
import sg2post
from pathlib import Path
OUT = Path(r"C:\work\farmai\집\클로드\research\audit_ec")
D = os.path.join(PK, "data")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)

z = np.load(os.path.join(PK, "output_14", "prediction_details.npz")); rid = z["row_id"]
s2 = np.load(os.path.join(PK, "output_14", "sg2_details.npz"))
x = pd.read_csv(os.path.join(D, "train_X.csv")); tx = pd.read_csv(os.path.join(D, "test_X.csv"))
y = pd.read_csv(os.path.join(D, "train_y.csv"))
yi = sg2post._ident(y); ec = yi.groupby(["farm", "day"]).sub_ec.mean(); ref = set(ec.index)
lo, hi = float(y.sub_ec.min()), float(y.sub_ec.max())
print("label range", lo, hi)
cols = ["row_id"] + sg2post.W + ["in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
S = sg2post.prepare(pd.concat([x[cols], tx[cols]], ignore_index=True)); cal = sg2post.ref_calendar(S, ref)
p13 = z["prediction"]
q = pd.DataFrame({"row_id": rid})
# instrumented copy of sg2post.correct
F = sg2post._ident(q).reset_index(drop=True); F["p"] = p13
WV, hrs, SIG, days = S["WV"], S["hrs"], S["SIG"], S["days"]
outp = F.p.values.copy(); log = []
cache = {}
def twin_date(f, d, h):
    E = [e for e in days[f] if (f, e) in ref]; cols_ = np.asarray(hrs <= h)
    A = WV.loc[[(f, e) for e in E]].values[:, cols_]; b = WV.loc[(f, d)].values[cols_]
    dist = np.sqrt(np.nanmean((A - b) ** 2, axis=1)); ok = dist <= .05
    return float(np.mean([cal[(f, e)] for e, o in zip(E, ok) if o])) if ok.any() else None
def full_date(f, d):
    if (f, d) in ref: return cal[(f, d)]
    if (f, d) in cache: return cache[(f, d)]
    t = twin_date(f, d, 23)
    if t is None:
        i = days[f].index(d); t = (full_date(f, days[f][i - 1]) + (0.0 if S["second"][(f, d)] else 0.1)) if i else 0.0
    cache[(f, d)] = t; return t
for (f, d), idx in F.groupby(["farm", "day"]).groups.items():
    E = [e for e in days[f] if (f, e) in ref and (f, e) in ec.index]; calc = np.array([cal[(f, e)] for e in E])
    i = days[f].index(d); prev = days[f][i - 1]
    rows = F.loc[idx].sort_values("hour"); cum = rows.p.expanding().mean().values
    for k, (ii, rr) in enumerate(rows.iterrows()):
        h = int(rr.hour); cq = twin_date(f, d, h) if h >= 5 else None; src = "twin"
        if cq is None: cq = full_date(f, prev) + 0.1; src = "prev"
        m = (np.abs(calc - cq) <= 3) & (calc != cq)
        rec = dict(row_id=rr.row_id, farm=f, day=d, hour=h, prev=prev, prev_gap=d - prev, src=src, cq=cq, ncand=int(m.sum()))
        if m.any():
            Em = [e for e, zz in zip(E, m) if zz]; Sh = SIG[h]
            RS = Sh.loc[[(f, e) for e in days[f] if (f, e) in ref]].astype(float)
            mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values
            qq = (Sh.loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(qq)
            C = np.nan_to_num(((Sh.loc[[(f, e) for e in Em]].values.astype(float) - mu) / sd)[:, use])
            dist = np.sqrt(((C - qq[use]) ** 2).mean(axis=1)) + .15 * np.abs(calc[m] - cq)
            j = int(np.argmin(dist)); a1 = float(ec[(f, Em[j])]); pm = cum[k]
            rec.update(best=Em[j], best_dist=float(dist[j]), a1=a1, pm=pm, applied=abs(a1 - pm) <= .30)
            if abs(a1 - pm) <= .30: outp[ii] = rr.p + 0.5 * (a1 - pm)
        log.append(rec)
L = pd.DataFrame(log)
sg = np.clip(outp, lo, hi)
print("SG2 instrumented == submitted SG2:", np.abs(sg - s2["pred_final"]).max())
print("SG2 rows: no candidates %d, guard-blocked %d, applied %d" % ((L.ncand == 0).sum(), (L.ncand.gt(0) & ~L.applied.fillna(False).astype(bool)).sum(), L.applied.fillna(False).astype(bool).sum()))
print("cand count dist:", L.ncand.describe().round(2).to_dict())
print("source of date: ", L.src.value_counts().to_dict())
R = L.groupby(["farm", "day"]).agg(prev=("prev", "first"), gap=("prev_gap", "first"), nc0=("ncand", lambda s: int((s == 0).sum())), ncand=("ncand", "median"),
                                   twin_hours=("src", lambda s: int((s == "twin").sum())), applied=("applied", lambda s: int(s.fillna(False).astype(bool).sum())),
                                   best23=("best", "last"), a1_23=("a1", "last"), pm23=("pm", "last"), cq23=("cq", "last"))
# calendar dates of best neighbour relative to query; best neighbour label & ID
R["best_dong_hint"] = R.best23
print(R.round(3).to_string())
# records whose best neighbour changes many times across hours
sw = L.dropna(subset=["best"]).groupby(["farm", "day"]).best.nunique()
print("distinct best neighbours per record over 24 h: ", sw.describe().round(2).to_dict())

# ---- EA2 anchors and final CSV
A = pd.read_csv(r"C:\work\farmai\집\클로드\research\local\linker_v7\eval60_confident_1step_anchor_v7.csv")
A2 = pd.read_csv(os.path.join(ROOT, r"온도\linker_v7\work\eval60_confident_1step_anchor_v7.csv")) if os.path.exists(os.path.join(ROOT, r"온도\linker_v7\work\eval60_confident_1step_anchor_v7.csv")) else None
print("anchor file rows", len(A), "package copy identical:", None if A2 is None else A.equals(A2))
G = pd.DataFrame({"row_id": rid, "ec14": s2["pred_final"], "p13": p13}); G["rec"] = G.row_id.str[:7]; G["hour"] = G.row_id.str[8:10].astype(int)
G = G.sort_values(["rec", "hour"]); cm = G.groupby("rec").ec14.transform(lambda s: s.expanding().mean())
AN = dict(zip(A.row, A.anchor_ec23)); a = G.rec.map(AN)
raw = np.where(a.notna(), G.ec14 + (a - cm), G.ec14)
G["ea2"] = np.clip(raw, lo, hi); G["clipped"] = (raw < lo) | (raw > hi)
e21 = pd.read_csv(os.path.join(PK, "output_21", "ec_ea2.csv")).set_index("row_id").sub_ec
s22 = pd.read_csv(os.path.join(ROOT, "submission_22.csv")).set_index("row_id").sub_ec
s22b = pd.read_csv(r"C:\work\farmai\제출\미제출\submission_22_측정감쇠앵커_v7EC앵커_가설검정\submission_22.csv").set_index("row_id").sub_ec
G = G.set_index("row_id")
print("audit EA2 vs output_21 max diff %.2e; output_21 vs submission_22 max diff %.2e; package vs 제출 copy %.2e" %
      (np.abs(G.ea2 - e21.loc[G.index]).max(), np.abs(e21 - s22.loc[e21.index]).max(), np.abs(s22 - s22b.loc[s22.index]).max()))
print("clipped rows by EA2:", int(G.clipped.sum()), "| rows at lo/hi in final:", int((s22 <= lo + 1e-6).sum()), int((s22 >= hi - 1e-6).sum()))
# 12th-round (submitted) EC for comparison
s19 = pd.read_csv(r"C:\work\farmai\제출\12회차_2026-10-11(나)\submission_19.csv").set_index("row_id").sub_ec
s14 = pd.read_csv(r"C:\work\farmai\제출\09회차_2026-10-06(팀)\submission_14.csv").set_index("row_id").sub_ec
print("9th submitted CSV vs ec14 max diff %.2e" % np.abs(s14.loc[G.index] - G.ec14).max())
G["s19"] = s19.loc[G.index]; G["s22"] = s22.loc[G.index]
# per record summary
P = G.reset_index().groupby("rec").agg(p13=("p13", "mean"), ec14=("ec14", "mean"), s19=("s19", "mean"), s22=("s22", "mean"))
P["anchor"] = P.index.map(AN)
P["farm"] = P.index.str[:3]; P["day"] = P.index.str[4:7].astype(int)
P["sg2_shift"] = P.ec14 - P.p13
print(P.round(3).to_string())
print("EA2 anchored records: mean shift s22 - ec14 = %.3f, mean |shift| %.3f" % ((P.s22 - P.ec14)[P.anchor.notna()].mean(), (P.s22 - P.ec14)[P.anchor.notna()].abs().mean()))
# training pass-2 daily mean distribution and same-dong (-2) jumps
td = ec.reset_index(); td.columns = ["farm", "day", "ec"]
t2 = td[td.day >= 179]
print("train pass-2 day mean EC: ", t2.ec.describe().round(3).to_dict())
print("train pass-1 day mean EC: ", td[td.day < 179].ec.describe().round(3).to_dict())
for c in ["p13", "ec14", "s22"]:
    print("eval day mean %s: " % c, P[c].describe().round(3).to_dict())
def jumps(df, col):
    o = []
    for f, g in df.groupby("farm"):
        g = g.set_index("day")[col]
        for d in g.index:
            if d - 2 in g.index: o.append(dict(farm=f, day=d, lag=2, j=g[d] - g[d - 2]))
            if d - 1 in g.index: o.append(dict(farm=f, day=d, lag=1, j=g[d] - g[d - 1]))
    return pd.DataFrame(o)
Jt = jumps(t2.rename(columns={"ec": "v"}), "v"); Je = jumps(P.rename(columns={"s22": "v"}), "v")
for lag in (1, 2):
    print("lag %d |jump| train p2 labels: median %.3f p90 %.3f | eval s22 preds: median %.3f p90 %.3f" %
          (lag, Jt[Jt.lag == lag].j.abs().median(), Jt[Jt.lag == lag].j.abs().quantile(.9), Je[Je.lag == lag].j.abs().median(), Je[Je.lag == lag].j.abs().quantile(.9)))
print("largest eval lag-2 jumps:\n", Je[Je.lag == 2].assign(a=lambda d: d.j.abs()).sort_values("a", ascending=False).head(8).to_string())
# training pass-2: how close is the day mean to the previous same-dong date's 23 h EC? (anchor plausibility) -- only label-level
yh = yi.set_index(["farm", "day", "hour"]).sub_ec
L.to_csv(OUT / "a3_sg2_log.csv", index=False); P.to_csv(OUT / "a3_records.csv"); G.to_csv(OUT / "a3_rows.csv")
