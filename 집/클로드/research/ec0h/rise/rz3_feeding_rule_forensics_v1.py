# -*- coding: utf-8 -*-
"""RZ3 (2026-10-12 집 클로드; 사용자: "EC 상승이 주기적이지 않다면 양액을 주는 조건이 있다는 말 같은데").  서술 포렌식, 학습 라벨만.
후보 규칙(출처: 수경재배 일반 지식, 대회 자료로 확인된 것 아님)과 실행 전 고정한 흔적:
 R1 일사 누적 관수: 희석(관수) 시작 시각이 일사에 따라 움직이고, 시작 시점의 '누적 일사'는 시계 시각보다 덜 흩어짐
    (판독: 희석 시작의 누적 일사 변동계수 < 시계 시각 변동계수 그리고 맑은 날 시작이 이름; 반대면 R2 타이머).
 R2 타이머 관수: 시작 시계 시각이 날씨와 무관하게 몰림.
 R3 '급액 EC 쪽으로 다가감'(관수 때 배지 EC → 급액 EC): 사건 직전 수준과 변화량이 음의 기울기(높을수록 더 내려감),
    변화=0 이 되는 수준 = 함의 급액 EC. 상승일은 함의 급액 EC 가 배지보다 높은 날 → 상승 모양이 '점점 작아지는 증가'(포화).
 R4 날씨 대응 처방(흐리거나 추우면 급액 EC 올림 — 일반 지식): 상승일 당일·전날짜(같은 농장 달력 −1일) 일사 낮음.
 R5 관수 감소로 농축: 일사 적은 날 오전 희석 작고 오후 상승 큼 — 전체 기록에서 일사와 오후 상승 음의 상관.
"""
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rng = np.random.default_rng(20261214)
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
TRi = np.where(TRAIN)[0]
E25 = EC / (1 + .02 * (ST - 25)); RAD = np.nan_to_num(X[:, :, 2]); CR = np.cumsum(RAD, 1)
A = pd.read_csv("rz2_records_v1.csv", index_col=0)
dE = np.diff(E25, axis=1)   # dE[:, h-1] = E(h) - E(h-1)
# ---- R1/R2: first dilution hour (6..16h, hourly drop <= -.03) ----
rows = []
for i in TRi:
    hs = [h for h in range(6, 17) if dE[i, h - 1] <= -.03]
    if hs:
        h = hs[0]; rows.append(dict(i=i, farm=FARM[i], h=h, cr=CR[i, h - 1], rad=CR[i, 23], lvl=E25[i, h - 1], drop=E25[i, min(h + 2, 23)] - E25[i, h - 1]))
Dl = pd.DataFrame(rows)
print("records with a morning/midday dilution (drop <= -.03/h in 6..16h): %d / %d" % (len(Dl), len(TRi)))
for f, G in Dl.groupby("farm"):
    G = G[G.rad > 0]
    cvh = G.h.std() / G.h.mean(); cvc = G.cr.std() / G.cr.mean()
    rho = pd.Series(G.h.values).corr(pd.Series(G.rad.values), method="spearman")
    print(f"  {f}: n {len(G)} | onset hour median {G.h.median():.0f} (IQR {G.h.quantile(.25):.0f}-{G.h.quantile(.75):.0f}), CV {cvh:.2f} | cum. radiation at onset median {G.cr.median():.0f}, CV {cvc:.2f} | Spearman(onset hour, day radiation) {rho:+.2f}")
    print("     onset hour counts:", G.h.value_counts().sort_index().to_dict())
    # within sunny vs dull tertile
    q = G.rad.quantile([1 / 3, 2 / 3]).values
    for nm, s in (("dull", G.rad <= q[0]), ("sunny", G.rad >= q[1])):
        print(f"     {nm:5s}: onset hour median {G[s].h.median():.1f}, cum-rad at onset median {G[s].cr.median():.0f}")
# ---- R3: approach to feed EC: hourly change vs level during active hours, per farm ----
print("\n[R3] hourly EC25 change vs previous level, hours 8..19, big moves |dE|>=.03 only:")
for f in ("F13", "F47"):
    I = TRi[FARM[TRi] == f]; L = E25[I, 7:19].ravel(); d = dE[I, 7:19].ravel(); m = np.isfinite(L) & np.isfinite(d) & (np.abs(d) >= .03)
    b, a = np.polyfit(L[m], d[m], 1)
    print(f"  {f}: n {m.sum()} | slope {b:+.3f} per unit level, zero-change level (implied feed EC) {-a/b:.2f} | down moves {(d[m]<0).sum()} median level {np.median(L[m][d[m]<0]):.2f}, up moves {(d[m]>0).sum()} median level {np.median(L[m][d[m]>0]):.2f}")
    for lo, hi in ((0, .5), (.5, .8), (.8, 1.2), (1.2, 2), (2, 9)):
        s = m & (L >= lo) & (L < hi)
        if s.sum() >= 10: print(f"     level {lo:.1f}-{hi:.1f}: big moves {s.sum():4d} | share up {(d[s] > 0).mean():.2f} | median move {np.median(d[s]):+.3f}")
# per-record implied target: on records with >=3 big moves, fit d = k*(T - L)
print("\n[R3b] per-record implied feed EC (records with >=3 big moves in 8..19h, k>0):")
rec = []
for i in TRi:
    L = E25[i, 7:19]; d = dE[i, 7:19]; m = np.isfinite(L) & np.isfinite(d) & (np.abs(d) >= .03)
    if m.sum() >= 3 and np.ptp(L[m]) > .02:
        b, a = np.polyfit(L[m], d[m], 1)
        if b < 0: rec.append(dict(i=i, farm=FARM[i], p2=NUM[i] >= 179, T=-a / b, k=-b, day=np.nanmean(E25[i]), up=A.at[i, "up"]))
Rc = pd.DataFrame(rec)
if len(Rc):
    Rc = Rc[(Rc["T"] > -1) & (Rc["T"] < 6)]
    print("  n %d | implied feed EC median %.2f (IQR %.2f-%.2f) | corr with day-mean EC25 %.2f | k median %.2f" % (len(Rc), Rc["T"].median(), Rc["T"].quantile(.25), Rc["T"].quantile(.75), Rc["T"].corr(Rc.day), Rc.k.median()))
    print("  rise days (up>=.15) implied T - day level median %+.2f vs others %+.2f" % ((Rc[Rc.up >= .15]["T"] - Rc[Rc.up >= .15].day).median(), (Rc[Rc.up < .15]["T"] - Rc[Rc.up < .15].day).median()))
# saturation shape on rise days: successive hourly increments in the rise window
R = A.index[A.up >= .15].values
inc = np.nanmean(dE[R, 12:21], 0)
print("\n[R3c] rise days mean hourly increments 13h..21h:", np.round(inc, 3).tolist(), "(decreasing after the peak = approach to a ceiling)")
# ---- R4/R5: weather on rise days, today and previous calendar day (same farm) ----
print("\n[R4/R5] radiation today and previous calendar date vs within-day change, pass 1 (calendar known):")
P1 = A[~A.p2 & A.cal.notna()].copy(); P1["d"] = np.round(P1.cal).astype(int)
dayrad = P1.groupby(["farm", "d"]).rad.median()
P1["rad_prev"] = [dayrad.get((f, d - 1), np.nan) for f, d in zip(P1.farm, P1.d)]
P1["rad_prev2"] = [np.nanmean([dayrad.get((f, d - k), np.nan) for k in (1, 2, 3)]) for f, d in zip(P1.farm, P1.d)]
for band, s in (("all", P1.lvl0 > -9), ("high (0h>=1)", P1.lvl0 >= 1), ("normal (<1)", P1.lvl0 < 1)):
    Z = P1[s]
    out = []
    for v in ("rad", "rad_prev", "rad_prev2", "otemp"):
        out.append(f"{v} ρ {Z.up.corr(Z[v], method='spearman'):+.2f}")
    print(f"  {band:13s} n {len(Z):3d} | Spearman with within-day change: " + ", ".join(out))
# R5: morning dilution size vs radiation (all records)
A["mdrop"] = [np.nanmin(E25[i, 8:14]) - np.nanmean(E25[i, 5:8]) for i in A.index]
A["aft"] = [np.nanmean(E25[i, 17:21]) - np.nanmin(E25[i, 10:16]) for i in A.index]
for band, s in (("all", A.lvl0 > -9), ("high", A.lvl0 >= 1), ("normal", A.lvl0 < 1)):
    Z = A[s]
    print(f"  [{band}] morning drop vs radiation ρ {Z.mdrop.corr(Z.rad, method='spearman'):+.2f} | afternoon rise vs radiation ρ {Z.aft.corr(Z.rad, method='spearman'):+.2f} | afternoon rise vs morning drop ρ {Z.aft.corr(Z.mdrop, method='spearman'):+.2f}")
