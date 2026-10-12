import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd, numpy as np, glob
f9 = glob.glob("09회차*/submission_14.csv")[0]; f12 = glob.glob("12회차*/submission_19.csv")[0]; f13 = glob.glob("13회차*/submission_22.csv")[0]
A = {k: pd.read_csv(f).set_index("row_id").sub_ec for k, f in (("9", f9), ("12", f12), ("13", f13))}
d12 = (A["12"] - A["9"]); d13 = (A["13"] - A["9"])
W = set(d12[d12.abs() > 1e-9].index.str[:7]); V = set(d13[d13.abs() > 1e-9].index.str[:7])
print("rows changed vs 9th: 12th", int((d12.abs() > 1e-9).sum()), "records", sorted(W)); print("13th", int((d13.abs() > 1e-9).sum()), "records", len(V))
print("overlap records:", sorted(W & V))
for r in sorted(W & V):
    m = A["9"].index.str[:7] == r
    print(f"  {r}: day mean 9th {A['9'][m].mean():.3f} | 12th(W2) {A['12'][m].mean():.3f} | 13th(v7) {A['13'][m].mean():.3f}")
print("RMS of change: W2 %.4f (on all 1440 rows), v7 %.4f" % (np.sqrt((d12**2).mean()), np.sqrt((d13**2).mean())))
d = (A["13"] - A["12"]); m = A["9"].index.str[:7].isin(list(W))
print("13 vs 12 on W2 records: max abs diff %.6f" % d[m].abs().max())
X = sorted(set(d[d.abs() > 1e-9].index.str[:7])); print("13 vs 12 changed records:", len(X), X)
s = (d ** 2).groupby(d.index.str[:7]).sum(); s = s[s > 0].sort_values(ascending=False)
mm = (d).groupby(d.index.str[:7]).mean()
E = pd.read_csv(r"C:\work\farmai\집\클로드\research\local\linker_v7\eval60_predecessor_v7.csv", encoding="utf-8-sig").set_index("row")
print("share of change^2 by record (13 vs 12), mean change, anchor prev, p, cross-ID:")
for r, v in s.items(): print(f"  {r} share {v / s.sum():.2f} mean {mm[r]:+.3f} | prev {E.at[r,'prev_h0']} p {E.at[r,'p_h0']:.2f} xf {bool(E.at[r,'xf'])}")
C = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); DAY = dict(zip(C.rec, C.day))
print("\ndate check (DM11): record day, prev day, gap")
for r in s.index:
    p = E.at[r, 'prev_h0']; print(f"  {r} day {DAY.get(r)} prev {p} day {DAY.get(p)} gap {DAY.get(r, np.nan) - DAY.get(p, np.nan)} share {s[r]/s.sum():.2f}")
