# -*- coding: utf-8 -*-
"""HS4 (2026-10-12 집 클로드; 사용자: '4번(고EC 동 안 크기와 입력) · 6번(고EC 급락일 흔적) 더 조사').  서술.
(A) 크기: 학습 고EC 기록(하루 평균 ≥ 1)마다 정답 하루 평균 y, 모델 잔차 r = y − q(DIAG10 OOF 3시드 평균).
    자기 입력 요약(하루 전체 / 0~5시 평균; 비인과 상한 먼저)과 y, r 의 순위상관(스피어만), 그리고 같은 동 에피소드 경과일(진단용 연결로 고EC 가 며칠째인지).
    판단: |ρ| ≥ .5 인 입력이 r 에 있으면 '크기 단서 후보'.  n 이 작아(≈35) 우연 기준선 = 입력 순서를 섞은 1000회의 최대 |ρ| 95% 분위.
(B) 급락: 진단용 같은 동 연결 b ← a 에서 a 하루 평균 ≥ 1 이고 하루 평균 변화 ≤ −0.3 인 날(급락) vs 고EC 가 이어진 날(변화 > −0.1).
    급락이 하루 중 언제 일어나나(정답 EC 시간별: 전날 23시 → 0시 차이, 0시 → 23시 차이), 급락일과 전날의 입력 변화(구동기·실내)를 고EC 지속일과 비교.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
here = os.path.dirname(os.path.abspath(__file__))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); X = X[X.row_id.str[:3].isin(["F13", "F47"])]
X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int)
V = ["out_temp", "out_hum", "out_rad", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
F = X.groupby("rec")[V].mean().join(X[X.h <= 5].groupby("rec")[V].mean().add_suffix("@0-5"))
F["rad_sum"] = X.groupby("rec").out_rad.sum()
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]; O["rec"] = O.row_id.str[:7]
q = O.groupby("rec").sg.mean()
y = pd.Series({r: float(np.nanmean(EC[idx[r]])) for r in F.index})
CAL = pd.read_csv(os.path.join(here, "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
TL = pd.read_pickle(S3 + r"\truth.pkl"); pre = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b)}
def age(r):
    k, x = 0, r
    while True:
        p = pre.get(x)
        if p is None or p not in y.index or y[p] < 1: return k
        k += 1; x = p
Hr = [r for r in y.index if y[r] >= 1 and r in q.index]
D = F.loc[Hr].copy(); D["y"] = y[Hr]; D["r"] = D.y - q[Hr]; D["age"] = [age(r) for r in Hr]; D["day"] = [DAY.get(r) for r in Hr]
rk = lambda s: s.rank().values
def rho(a, b):
    m = np.isfinite(a) & np.isfinite(b); return np.corrcoef(rk(pd.Series(a[m])), rk(pd.Series(b[m])))[0, 1] if m.sum() > 5 else np.nan
cols = list(F.columns) + ["age", "day"]
rng = np.random.default_rng(3)
null = [max(abs(rho(D[c].values, rng.permutation(D.r.values))) for c in cols if D[c].nunique() > 2) for _ in range(300)]
thr = float(np.quantile(null, .95))
R = pd.DataFrame([dict(v=c, rho_y=rho(D[c].values, D.y.values), rho_resid=rho(D[c].values, D.r.values)) for c in cols if D[c].nunique() > 2]).set_index("v")
pd.set_option("display.width", 200)
print(f"(A) high-EC training records with OOF: {len(D)} | mean y {D.y.mean():.2f}, mean residual {D.r.mean():+.2f} | chance max |rho| 95% = {thr:.2f}")
print(R.reindex(R.rho_resid.abs().sort_values(ascending=False).index).head(12).round(2).to_string())
print("\n   residual by episode age:", D.groupby(pd.cut(D.age, [-1, 0, 2, 5, 30])).r.agg(["count", "mean"]).round(2).to_dict("index"))
# (B) drops
rows = []
for b, a in pre.items():
    if a in y.index and b in y.index and y[a] >= 1 and DAY.get(b) == DAY.get(a, -9) + 1:
        ia, ib = idx[a], idx[b]
        rows.append(dict(a=a, b=b, ya=y[a], yb=y[b], dmean=y[b] - y[a], jump_mid=EC[ib, 0] - EC[ia, 23], within_b=EC[ib, 23] - EC[ib, 0], within_a=EC[ia, 23] - EC[ia, 0], min_hour_b=int(np.nanargmin(EC[ib]))))
B = pd.DataFrame(rows); B["kind"] = np.select([B.dmean <= -.3, B.dmean > -.1], ["DROP", "persist"], "mid")
print(f"\n(B) same-house day pairs starting high: {len(B)} | DROP {int((B.kind=='DROP').sum())}, persist {int((B.kind=='persist').sum())}")
print(B[B.kind == "DROP"].round(2).to_string(index=False))
print("   persist: mean within-day change a %.2f / b %.2f, midnight jump %.2f" % (B[B.kind == "persist"].within_a.mean(), B[B.kind == "persist"].within_b.mean(), B[B.kind == "persist"].jump_mid.mean()))
dif = []
for _, r_ in B[B.kind != "mid"].iterrows():
    d = F.loc[r_.b] - F.loc[r_.a] if (r_.b in F.index and r_.a in F.index) else None
    if d is not None: dif.append(pd.concat([d, pd.Series({"kind": r_.kind})]))
Dd = pd.DataFrame(dif)
num = [c for c in Dd.columns if c != "kind"]
G = Dd.groupby("kind")[num].apply(lambda g: g.astype(float).median()).T
G["gap"] = G.get("DROP", np.nan) - G.get("persist", np.nan)
print("\n   input change (drop day minus previous day), median; DROP vs persist:")
print(G.reindex(G.gap.abs().sort_values(ascending=False).index).head(12).round(2).to_string())
