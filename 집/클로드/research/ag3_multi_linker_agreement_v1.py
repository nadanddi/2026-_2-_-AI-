# -*- coding: utf-8 -*-
"""AG3 1단계 (2026-10-14 집 클로드; 2차 비평 2·3순위).  서술(판정 아님; 진행 조건 미리 정함).  온도 앵커용 링커 일치.
TNA(v15R 1순위 == HP2) 는 거울 정확도 1.000 대 → 14회차 TNA8 이 평가에서 성립.  일치 원천을 넓히면 정밀도 유지하며 앵커가 늘까?
링커(모두 거울 기하 = tb2 mirror_lab 가시성; 평가는 학습 정답 전체 가시):
  V15 = v15R 1순위(prev0; 거울 v15r15_mirror / 평가 v15r15_eval).
  HP2 = tb2/td8 hp2(DMT2 날짜 d−1 + 지문).
  GPRS = GP-RS 규정판(6.838): v15R 후보 순서대로, 후보 a 를 '보이는 정답 학습 기록 b′ 가 1순위로 더 높은 확률로 지목' 하면 제외 → 첫 남은 후보.
  LK0 = LKH1 0시판: DMT3e 날짜 d−1·같은 구간의 보이는 정답 학습 후보 중 0시 운영 지문(구동기·CO2 8열, 학습 기록만으로 표준화) 최근접, 2등/1등 >= 1.5.
일치 쌍(GP-RS 는 v15R 에서 나와 독립 아님 → GPRS 가 V15 와 다를 때만 셈): HP2&GPRS*, V15&LK0, HP2&LK0, GPRS*&LK0.
대상 = 정답 있는 진짜 전날 기록이 보이는 학습 기록 중 v7 앵커·TNA8 앵커 아닌 것(새로 붙을 수 있는 기록).  정확도 = CLEAN(진단 연결, 달력 차 1) 대비.
진행 조건(미리 정함): 어떤 쌍이 거울에서 정확도 >= .98(전체·2차 각각, 2차 n >= 3) & 평가에서 v23 밖 새 기록 >= 2 → 2단계(TB3) 판정.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "tb2_temp_agree_any_p_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "tb2_temp_agree_any_p_v1.py")
exec(src[:src.index("DD = pd.read_pickle")])
__file__ = os.path.join(HERE, "ag3_multi_linker_agreement_v1.py")
C3 = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); D3 = dict(zip(C3.rec, C3.day)); S3d = dict(zip(C3.rec, C3.seg))
FP0 = np.stack([_X[_X.h == 0].set_index("rec")[v].reindex([rid(i) for i in range(N)]).values for v in _V], 1).astype(float)
FP0 = (FP0 - np.nanmean(FP0[TRAIN], 0)) / (np.nanstd(FP0[TRAIN], 0) + 1e-9)
def tdate(b, vis):
    vi = np.where(vis)[0]; vi = vi[vi != b]; m = vi[np.all(np.abs(OUTX[vi, 0] - OUTX[b, 0]) < 1e-6, axis=1)]; m = [i for i in m if rid(i) in D3]
    if not m: return None
    d = pd.Series([D3[rid(i)] for i in m]).mode().iloc[0]; return d, pd.Series([S3d[rid(i)] for i in m if D3[rid(i)] == d]).mode().iloc[0]
def lk0(b, vis):
    t = tdate(b, vis)
    if t is None: return None
    cand = [i for i in np.where(vis)[0] if i != b and D3.get(rid(i)) == t[0] - 1 and S3d.get(rid(i)) == t[1]]
    if not cand: return None
    dd = sorted((float(np.nanmean(np.minimum(np.abs(FP0[i] - FP0[b]), 4))), i) for i in cand)
    return rid(dd[0][1]) if (len(dd) == 1 or dd[1][0] >= 1.5 * max(dd[0][0], 1e-9)) else None
def gprs(P, r, vis):
    row = P[P.rec == r]
    if not len(row): return None
    q = row.iloc[0]; cands = [(q[f"prev{j}"], float(q[f"p{j}"])) for j in range(6) if isinstance(q.get(f"prev{j}"), str)]
    for a, p in cands:
        rival = P[(P.prev0 == a) & (P.rec != r) & (P.p0 > p) & P.rec.map(lambda x: x in idx and bool(vis[idx[x]]))]
        if len(rival) == 0: return a
    return None
def picks(P15, r, vis):
    b = idx[r]; row = P15[P15.rec == r]; v15 = row.prev0.iloc[0] if len(row) and isinstance(row.prev0.iloc[0], str) else None
    g = gprs(P15, r, vis); return dict(V15=v15, HP2=hp2(b, vis), GPRS=g if g != v15 else None, LK0=lk0(b, vis))
PAIRS = [("HP2", "GPRS"), ("V15", "LK0"), ("HP2", "LK0"), ("GPRS", "LK0")]
TN8M = ARMS["TNA8"][1]; rows = []
for r in PM15.rec:
    b = idx[r]; true = CLEAN.get(r); vis = mirror_lab(r) & np.isfinite(ST[:, 23])
    if r in v7set or r in TN8M or true is None or not vis[idx[true]]: continue
    pk = picks(PM15, r, vis); d = dict(rec=r, p2=NUM[b] >= 179, true=true, **pk)
    for a, c in PAIRS: d[f"{a}&{c}"] = pk[a] if (pk[a] is not None and pk[a] == pk[c]) else None
    rows.append(d)
T = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(f"mirror targets (labelled true prev visible, not v7/TNA8): {len(T)} (pass-2 {int(T.p2.sum())})")
for k in ["V15", "HP2", "GPRS", "LK0"] + [f"{a}&{c}" for a, c in PAIRS]:
    for nm, g in (("all", T), ("p2", T[T.p2])):
        f = g[g[k].notna()]; print(f"  {k:10s} {nm:3s}: fires {len(f):3d} acc {np.mean(f[k] == f.true) if len(f) else float('nan'):.3f}")
# eval
E15 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_eval.csv")); E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig")
v7e = set(E7[(E7.p_h0 >= .8) & E7.prev_is_labelled.astype(bool) & (E7.dT_air.abs() <= 1.5)].row)
tna8e = {**v15_new(E15, lambda r: TRAIN.copy(), v7e, 0.0, True), **v15_new(E15, lambda r: TRAIN.copy(), v7e, .8, False)}
visE = TRAIN & np.isfinite(ST[:, 23]); ok_pairs = []
for a, c in PAIRS:
    k = f"{a}&{c}"
    for nm, g in (("all", T), ("p2", T[T.p2])):
        pass
    fa = T[T[k].notna()]; f2 = fa[fa.p2]
    acc_a = np.mean(fa[k] == fa.true) if len(fa) else 0; acc_2 = np.mean(f2[k] == f2.true) if len(f2) >= 3 else 0
    if acc_a >= .98 and acc_2 >= .98: ok_pairs.append(k)
newE = {}
for i in np.where(~TRAIN)[0]:
    r = rid(i)
    if r in v7e or r in tna8e: continue
    pk = picks(E15, r, visE)
    for a, c in PAIRS:
        if pk[a] is not None and pk[a] == pk[c]: newE.setdefault(f"{a}&{c}", {})[r] = pk[a]
print("\neval new records beyond v23 per pair:", {k: dict(sorted(v.items())) for k, v in newE.items()})
go = [k for k in ok_pairs if len(newE.get(k, {})) >= 2]
print("pairs meeting precision condition:", ok_pairs, "| go to stage 2:", go if go else "none -> stop AG3")
T.to_csv(os.path.join(HERE, "ag3_mirror_picks_v1.csv"), index=False)
