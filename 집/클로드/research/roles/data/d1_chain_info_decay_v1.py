# -*- coding: utf-8 -*-
"""역할: 데이터 구조 전문가 (2026-10-12 집 클로드 하위 에이전트). 진단·서술 전용(채택 판정 아님).
D1: 진실 사슬(linker_v3 truth a1, cert 여부 표시)을 따라 k단계 뒤 같은 동 정보가 대상 기록 0시 EC 잔차를 얼마나 설명하는가.
  r0 = y_t,0 - q_t,0 (DIAG10 CUR sg 3시드 평균 OOF)
  정보원 k: (i) 정답 앵커 차이  g_k = y_{a_k,23} - q_t,0  (k단계 전 같은 동 23시 정답; 중간 날 정답 없음 가정)
            (ii) 모델 잔차 지속 e_k = y_{a_k,23} - q_{a_k,23}  (k단계 전 모델 23시 잔차)
            (iii) 소프트 앵커 s_1 = q_{a_1,23} - q_t,0 (정답 불필요)
  지표: 상관, 원점 통과 기울기, 5겹(사슬 성분 묶음) CV R2, 그 정보로 보정했을 때 0시 RMSE 변화.
D2: 같은 소프트 앵커 신호를 EV(v2_ev_ec_rows, BASE EA2/R3, 같은 fold 진실 쌍)와 DIAG10에서 비교 + DIAG10 fold 관계(a1과 t가 같은 fold인지) 분해.
D3: 진실 같은 ID 전날의 번호 차 분포(1차/2차) — 번호 구조 사전정보가 얼마나 날카로운가."""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env, pickle
import numpy as np, pandas as pd
H = r"C:\work\farmai\집\클로드\research"
T = pd.read_pickle(os.path.join(H, "local/linker_v3/truth.pkl"))
r = pickle.load(open(os.path.join(H, "local/linker_v3/base.pkl"), "rb"))["recs"].reset_index(drop=True)
ids = np.array([f"{f}_{d:03d}" for f, d in zip(r.farm, r.day)])
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]; O["rec"] = O.row_id.str[:7]
q = O.groupby(["rec", "hour"]).sg.mean().unstack(); y = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
fold = O.groupby("rec").fold.first()
prev = {}; cert = {}
for _, z in T.iterrows():
    prev[ids[int(z.b)]] = ids[int(z.a1)]; cert[ids[int(z.b)]] = bool(z.cert)
recs = [t for t in q.index if t in prev]
def chain(t, k):
    a = t
    for _ in range(k):
        a = prev.get(a)
        if a is None or a not in q.index or a == t: return None
    return a
def comp(t):  # 사슬 성분 대리: 5단계 뒤 조상의 이름(같은 줄 묶음)
    a = t
    for _ in range(30):
        b = prev.get(a)
        if b is None or b not in q.index or b == t: break
        a = b
    return a
def cvr2(x, yv, g):
    ug = np.unique(g); rng = np.random.default_rng(0); fo = {u: i % 5 for i, u in enumerate(rng.permutation(ug))}
    f = np.array([fo[u] for u in g]); pr = np.zeros_like(yv)
    for k in range(5):
        tr = f != k; b = (x[tr] * yv[tr]).sum() / (x[tr] ** 2).sum(); pr[~tr] = b * x[~tr]
    return 1 - ((yv - pr) ** 2).sum() / (yv ** 2).sum(), np.sqrt(np.mean((yv - pr) ** 2)) / np.sqrt(np.mean(yv ** 2)) - 1
print("== D1 진실 사슬 k단계 정보 감쇠 (EC 0시 잔차, DIAG10 OOF) ==")
for part, cond in (("전체", lambda s: True), ("1차", lambda s: int(s[4:7]) < 179), ("2차", lambda s: int(s[4:7]) >= 179)):
    for k in range(1, 7):
        rows = []
        for t in recs:
            if not cond(t): continue
            a = chain(t, k)
            if a is None: continue
            rows.append((t, a, y.at[t, 0] - q.at[t, 0], y.at[a, 23] - q.at[t, 0], y.at[a, 23] - q.at[a, 23], all(cert.get(chain(t, j) if j else t, False) for j in range(k)), comp(t)))
        if len(rows) < 8: continue
        D = pd.DataFrame(rows, columns=["t", "a", "r0", "g", "e", "allcert", "comp"])
        r2g, dg = cvr2(D.g.values, D.r0.values, D.comp.values); r2e, de = cvr2(D.e.values, D.r0.values, D.comp.values)
        Dc = D[D.allcert]
        r2gc = cvr2(Dc.g.values, Dc.r0.values, Dc.comp.values)[0] if len(Dc) >= 15 else np.nan
        print(f"  {part} k={k}: n={len(D):3d}(cert {len(Dc):3d}) | 0시 잔차 RMS {np.sqrt((D.r0**2).mean()):.3f} | 정답앵커차 corr {np.corrcoef(D.g, D.r0)[0,1]:+.2f} CV-R2 {r2g:+.2f} (cert {r2gc:+.2f}) RMSE {100*dg:+.0f}% | 모델잔차지속 corr {np.corrcoef(D.e, D.r0)[0,1]:+.2f} CV-R2 {r2e:+.2f}")
print("== D2 소프트 앵커 s1 = q_{a1,23} - q_t,0 (정답 불필요) ==")
rows = []
for t in recs:
    a = chain(t, 1)
    if a is None: continue
    rows.append((t, a, y.at[t, 0] - q.at[t, 0], q.at[a, 23] - q.at[t, 0], y.at[a, 23] - q.at[t, 0], fold[t] == fold[a], t[:3] == a[:3], cert[t]))
D = pd.DataFrame(rows, columns=["t", "a", "r0", "s", "g", "samefold", "sameid", "cert"])
for nm, M in (("전체", D), ("DIAG10 같은 fold", D[D.samefold]), ("DIAG10 다른 fold", D[~D.samefold]), ("같은 ID", D[D.sameid]), ("다른 ID", D[~D.sameid]),
              ("1차", D[D.t.str[4:7].astype(int) < 179]), ("2차", D[D.t.str[4:7].astype(int) >= 179])):
    if len(M) < 5: continue
    b = (M.s * M.r0).sum() / (M.s ** 2).sum()
    print(f"  DIAG10 {nm}: n={len(M)} corr(s,r0) {np.corrcoef(M.s, M.r0)[0,1]:+.2f} 기울기 {b:+.2f} | 비교 corr(g,r0) {np.corrcoef(M.g, M.r0)[0,1]:+.2f} | sd(s) {M.s.std():.3f} sd(r0) {M.r0.std():.3f}")
# EV
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
F = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv")); F = F[F.role == "val"].drop_duplicates("rec").set_index("rec")
for col in ("R3", "EA2"):
    P = R.groupby(["rec", "hour"])[col].mean().unstack(); Y = R.groupby(["rec", "hour"]).y.first().unstack(); anc = R.groupby("rec").anchored.first()
    rows = []
    for t in P.index:
        a = prev.get(t)
        if a is None or a not in P.index: continue
        rows.append((t, a, Y.at[t, 0] - P.at[t, 0], P.at[a, 23] - P.at[t, 0], F.at[t, "fold"] == F.at[a, "fold"], bool(anc[t]), bool(anc[a]), t[:3] == a[:3]))
    E = pd.DataFrame(rows, columns=["t", "a", "r0", "s", "samefold", "anc_t", "anc_a", "sameid"])
    for nm, M in (("전체", E), ("t 앵커없음", E[~E.anc_t]), ("t·a 앵커없음", E[~E.anc_t & ~E.anc_a]), ("t 앵커없음·a 앵커있음", E[~E.anc_t & E.anc_a]),
                  ("같은 fold", E[E.samefold & ~E.anc_t]), ("다른 fold", E[~E.samefold & ~E.anc_t])):
        if len(M) < 5: continue
        b = (M.s * M.r0).sum() / (M.s ** 2).sum()
        print(f"  EV {col} {nm}: n={len(M)} corr {np.corrcoef(M.s, M.r0)[0,1]:+.2f} 기울기 {b:+.2f} sd(s) {M.s.std():.3f} sd(r0) {M.r0.std():.3f}")
# 같은 쌍 DIAG10 vs EV (t 앵커없음)
    if col == "EA2":
        J = E[~E.anc_t].merge(D[["t", "r0", "s"]], on="t", suffixes=("_ev", "_dg"))
        print(f"  [공통 쌍 n={len(J)}] EV corr {np.corrcoef(J.s_ev, J.r0_ev)[0,1]:+.2f} | DIAG10 corr {np.corrcoef(J.s_dg, J.r0_dg)[0,1]:+.2f} | r0 RMS EV {np.sqrt((J.r0_ev**2).mean()):.3f} DIAG {np.sqrt((J.r0_dg**2).mean()):.3f} | corr(r0_ev,r0_dg) {np.corrcoef(J.r0_ev, J.r0_dg)[0,1]:+.2f}")
print("== D3 진실 전날 번호 차(같은 ID) ==")
for part, lo, hi in (("1차", 0, 179), ("2차", 179, 999)):
    gaps = [int(t[4:7]) - int(prev[t][4:7]) for t in prev if lo <= int(t[4:7]) < hi and prev[t][:3] == t[:3]]
    xid = sum(1 for t in prev if lo <= int(t[4:7]) < hi and prev[t][:3] != t[:3]); n = sum(1 for t in prev if lo <= int(t[4:7]) < hi)
    print(f"  {part}: 기록 {n}, 다른 ID 전날 {xid} ({100*xid/n:.0f}%), 같은 ID 번호차 분포 {pd.Series(gaps).value_counts().sort_index().to_dict()}")
