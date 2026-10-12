# DF1b (DF1 사후 서술): DF 정확도가 LF1(82%)보다 낮은 이유 — 진짜 전날이 후보(보이는 정답 d−1 기록)에 들어 있었나.
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v9")
from feat9 import *
P = pd.read_csv(os.path.join(env.LOCAL, "linker_v9", "df1_picks_mirror.csv"))
V7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig").set_index("rec")
P = P[(~P.v7_anchored) & (P.truth.fillna("") != "")].copy()
CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); DAY = dict(zip(CAL.rec, CAL.day))
idx = {rid(i): i for i in range(N)}
def in_cand(row):
    # truth visible-labelled and on d-1 : truth labelled in reality AND not hidden in this mirror chunk -> approximated by V7 prev_is_labelled semantics: hidden if same chunk
    t = row.truth; b = idx[row.rec]; ti = idx[t]; f = FARM[b]; s = (NUM[b] - (NUM[FARM == f].min())) // 10 * 10 + NUM[FARM == f].min()
    g = "F47" if f == "F13" else "F13"; m = s + (-2 if f == "F13" else 2)
    hidden = (FARM[ti] == f and s - 1 <= NUM[ti] <= s + 10) or (FARM[ti] == g and m - 1 <= NUM[ti] <= m + 10)
    return bool(TRAIN[ti]) and not hidden and DAY.get(t) == DAY.get(row.prev)
P["truth_in_cand"] = P.apply(in_cand, axis=1)
for k, g in P.groupby("truth_in_cand"):
    print(f"truth among candidates = {k}: n {len(g)} | accuracy {g.correct.mean():.3f} | margin>=.1 n {int((g.margin>=.1).sum())} acc {g[g.margin>=.1].correct.mean():.3f} | single-candidate n {int((g.ncand==1).sum())}")
fpd = lambda b, a: float(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0]), 4)))
P["d_best"] = [fpd(idx[r], idx[p]) for r, p in zip(P.rec, P.prev)]
print("\nabsolute fingerprint distance of the chosen candidate (hour 0):")
print(P.groupby("truth_in_cand").d_best.describe().round(3).to_string())
for t in (0.05, 0.08, 0.10, 0.15, 0.20):
    S = P[P.d_best <= t]; print(f"   d_best <= {t:.2f}: n {len(S)}, accuracy {S.correct.mean() if len(S) else float('nan'):.3f}, truth-in-cand share {S.truth_in_cand.mean() if len(S) else float('nan'):.2f}")
