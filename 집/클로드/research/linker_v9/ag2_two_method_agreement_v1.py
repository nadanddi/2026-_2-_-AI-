# AG2 (2026-10-12 집 클로드; 사용자: '애매한 날짜·동을 다른 모델이 확인해 개선 가능?') 서술: 짝찾기 v7(자정 연속·달력)과 DF(날짜 먼저 + 0시 지문)가
# 같은 전날을 고르면 맞을 확률이 얼마나 되나 — 거울 CV, 진실 알려진 기록.  '확인 모델' 아이디어의 신호 크기 측정.
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v9")
from feat9 import *
idx = {rid(i): i for i in range(N)}
fpd = lambda b, a: float(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0]), 4)))
P = pd.read_csv(os.path.join(env.LOCAL, "linker_v9", "df1_picks_mirror.csv")).set_index("rec"); P["truth"] = P.truth.fillna("")
V = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig").set_index("rec")
J = V.join(P[["prev", "margin", "ncand", "truth"]].rename(columns={"prev": "df_prev"}), how="left")
TP = {rid(int(b)): rid(int(a)) for a, b in zip(pd.read_pickle(S3 + r"\truth.pkl").a1, pd.read_pickle(S3 + r"\truth.pkl").b)}
J["truth"] = [TP.get(r, "") for r in J.index]; J = J[J.truth != ""]
J["agree"] = J.prev == J.df_prev; J["v7_ok"] = J.prev == J.truth; J["lab"] = J.prev_is_labelled.astype(bool)
J["d_best"] = [fpd(idx[r], idx[p]) if isinstance(p, str) else np.nan for r, p in zip(J.index, J.df_prev)]
anc = (J.p >= .8) & J.lab
print(f"records with truth: {len(J)} | v7 anchored (p>=.8, labelled) {int(anc.sum())} acc {J[anc].v7_ok.mean():.3f}")
for nm, S in (("v7 anchored", J[anc]), ("v7 NOT anchored but chose labelled", J[~anc & J.lab])):
    for a, g in S.groupby("agree"):
        print(f"  {nm:36s} DF agrees={a}: n {len(g):3d}, v7 choice correct {g.v7_ok.mean():.3f}")
S = J[~anc & J.lab & J.agree]
for lo in (.5, .6, .7):
    g = S[S.p >= lo]; print(f"  not anchored, labelled, DF agrees, p >= {lo}: n {len(g)}, correct {g.v7_ok.mean() if len(g) else float('nan'):.3f}, pass-2 {int((g.index.str[4:7].astype(int) >= 179).sum())}")
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig").set_index("row")
N4 = pd.read_csv(r"C:\work\farmai\집\클로드\research\na4_v1.csv").set_index("rec")
print("\neval unanchored where v7 labelled choice == fingerprint-best (NA4 chose_is_fpbest):", N4[N4.chose_is_fpbest].index.tolist())
