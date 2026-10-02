# -*- coding: utf-8 -*-
"""Claude independent check of Codex's submission_10 EC delivery (2026-10-02)."""
import env, os, sys, json, importlib.util, hashlib, numpy as np, pandas as pd
A = os.path.join(env.ROOT, u"집", u"코덱스", "local", "ec_submission10_season_20261002_v2", "artifact")
man = json.load(open(os.path.join(A, "delivery_manifest.json"), encoding="utf-8"))
print("0. delivery_manifest status:", man.get("status"))
sub = pd.read_csv(os.path.join(A, "submission_10.csv"), dtype={"sub_temp": str})
ss = pd.read_csv(os.path.join(env.DATA, "sample_submission.csv"))
print("1. columns", list(sub.columns), "rows", len(sub), "| row_id order == sample_submission:", sub.row_id.tolist() == ss.row_id.tolist(),
      "| duplicates", sub.row_id.duplicated().sum())
print("   sub_temp all blank:", sub.sub_temp.isna().all() or (sub.sub_temp.fillna("") == "").all(),
      "| sub_ec finite:", np.isfinite(sub.sub_ec).all(), "range %.3f..%.3f mean %.3f" % (sub.sub_ec.min(), sub.sub_ec.max(), sub.sub_ec.mean()))
# 2. independent season for the 60 eval days with Claude's DC4 code, ALL labelled training days (400, lock included)
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc4", "ec2_DC4_exact_twin_anchor_v1.py"); dc4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc4)
tX = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); ty = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
f = tX[tX.row_id.str[:3].isin(["F13", "F47"])]
wv = dc4.weather_vectors(f)
y = ty[ty.row_id.str[:3].isin(["F13", "F47"]) & ty.sub_ec.notna()]
p = y.row_id.str.split("_", expand=True); tr = pd.DataFrame({"farm": p[0], "day": p[1].astype(int)}).drop_duplicates()
q = ss.row_id.str.split("_", expand=True); te = pd.DataFrame({"farm": q[0], "day": q[1].astype(int)}).drop_duplicates().reset_index(drop=True)
season, te["season_claude"] = dc4.season_index(tr, te, wv)
ev = pd.read_csv(os.path.join(A, "stage", "evaluation_day_to_season.csv")) if os.path.exists(os.path.join(A, "stage", "evaluation_day_to_season.csv")) else pd.read_csv(os.path.join(os.path.dirname(A), "..", "..", "analysis", "ec_submission10_season_20261002_v1", "delivery_ec_only_v1", "evaluation_day_to_season.csv"))
print("   codex eval season table columns:", list(ev.columns)[:6], "rows", len(ev))
sc = [c for c in ev.columns if "season" in c.lower()][0]
m = te.merge(ev[["farm", "day", sc]], on=["farm", "day"])
print("2. eval days matched %d/60, max |claude - codex season| %.3e" % (len(m), np.abs(m.season_claude - m[sc]).max()))
trs = pd.read_csv(os.path.join(A, "stage", "training_day_to_season.csv")) if os.path.exists(os.path.join(A, "stage", "training_day_to_season.csv")) else None
if trs is not None:
    tc = [c for c in trs.columns if "season" in c.lower()][0]
    trs["claude"] = [season.get((a, b), np.nan) for a, b in zip(trs.farm, trs.day)]
    print("   training days %d, max |diff| %.3e" % (len(trs), np.nanmax(np.abs(trs.claude - trs[tc]))))
# 3. change vs round-3 EC by eval calendar group (diagnostic only)
p3 = pd.read_csv(env.submission_path("submission_04.csv")).set_index("row_id").reindex(sub.row_id)
cal = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_9_days.csv"))[["farm", "day", "cal"]].set_index(["farm", "day"]).cal
c = np.array([cal.get((a, int(b)), np.nan) for a, b in zip(q[0], q[1])])
d = sub.sub_ec.values - p3.sub_ec.values
print("3. vs round-3 EC: RMS change %.4f, mean %+.4f" % (np.sqrt((d ** 2).mean()), d.mean()))
for nm, mm in (("eval cal<70", c < 70), ("eval cal>=70", c >= 70)):
    print("   %-13s rows %d: round3 mean %.3f -> sub10 mean %.3f (change %+.3f)" % (nm, mm.sum(), p3.sub_ec.values[mm].mean(), sub.sub_ec.values[mm].mean(), d[mm].mean()))
blk = pd.DataFrame({"farm": q[0], "day": q[1].astype(int), "r3": p3.sub_ec.values, "s10": sub.sub_ec.values}).groupby(["farm", "day"]).mean().round(3)
print(blk.reset_index().assign(cal=[cal.get((a, b)) for a, b in zip(blk.reset_index().farm, blk.reset_index().day)]).to_string(index=False))
print("sha256 submission_10.csv:", hashlib.sha256(open(os.path.join(A, "submission_10.csv"), "rb").read()).hexdigest())
