# -*- coding: utf-8 -*-
"""RZ2 (2026-10-12 집 클로드).  RZ1 의 한계: 확실 연결(cert)은 자정 EC 연속으로 만든 것이라 상승일이 거의 빠짐(150쌍 중 +.15 이상 5개).
그래서 연결 없이 '하루 안 상승'으로 본다.  기록 r 의 상승량 up = mean EC25(20..23) - mean EC25(0..3).  (학습 라벨, 서술 조사)
달력: 1차(번호<179)는 feat4.calendar 의 F13 번호 척도(F47 은 같은 날짜 묶음으로 맞춘 사상), 2차는 따로(번호 그대로).
사전 판독: (a) 주기: 농장별 달력 위치 정수일 기준 상승 기록 비율의 위상 집중(P=2..21, 순열, 본페로니 .0025)과 상승일 간 간격;
(b) 같은 날짜 다른 동 동시 상승(묶음 안 up 상관 vs 무작위); (c) 상승 시각; (d) 상승일 조건: 0시 EC25 수준, 외기 일사·외기온,
실내온, 천창·차광·유동팬·CO₂ 하루 평균 — 달력 ±7일 같은 농장 기록 대비 차이(계절 제거), AUC; (e) 상승 다음 날(확실 연결이 있으면) 지속.
"""
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
from collections import Counter
rng = np.random.default_rng(20261213)
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
TRi = np.where(TRAIN)[0]
vis = TRAIN.copy(); comp = components(vis); gpos, mp = calendar(vis, comp)
E25 = EC / (1 + .02 * (ST - 25))
A = pd.DataFrame(dict(i=TRi, rec=[rid(i) for i in TRi], farm=FARM[TRi], num=NUM[TRi], p2=NUM[TRi] >= 179, comp=[comp[i] for i in TRi])).set_index("i")
A["cal"] = [gpos.get(comp[i], mp(i)) if NUM[i] < 179 else np.nan for i in TRi]
A["up"] = np.nanmean(E25[TRi, 20:24], 1) - np.nanmean(E25[TRi, 0:4], 1)
A["lvl0"] = np.nanmean(E25[TRi, 0:2], 1); A["day"] = np.nanmean(E25[TRi], 1)
hrs = np.arange(24)
def onset(i):
    e = E25[i]; tot = np.nanmean(e[20:24]) - np.nanmean(e[0:4])
    if not np.isfinite(tot) or tot <= 0: return np.nan
    c = e - np.nanmean(e[0:4]); k = np.where(c >= tot / 2)[0]; return float(k[0]) if len(k) else np.nan
A["onset"] = [onset(i) for i in TRi]
IV = dict(rad=X[:, :, 2], otemp=X[:, :, 0], tin=IN_[:, :, 0], hum=IN_[:, :, 1], co2=IN_[:, :, 2], vent=AC[:, :, 0], shade=AC[:, :, 1],
          thermal=AC[:, :, 2], heat=AC[:, :, 3], fan=AC[:, :, 4], st=ST)
for k, v in IV.items(): A[k] = np.nanmean(v[TRi], 1) if k != "rad" else np.nansum(v[TRi], 1)
print("records", len(A), "| up quantiles:", np.round(A.up.quantile([.05, .25, .5, .75, .9, .95, .99]).values, 3).tolist())
print("down share up<=-.10 %.2f, up>=.10 %.2f, >=.20 %.2f" % ((A.up <= -.1).mean(), (A.up >= .1).mean(), (A.up >= .2).mean()))
for TH in (.10, .15, .20):
    R = A.up >= TH
    print(f"\n===== within-day rise >= +{TH:.2f} : {int(R.sum())} records (pass-1 {int((R & ~A.p2).sum())}, pass-2 {int((R & A.p2).sum())}) =====")
    print("  by farm:", A[R].farm.value_counts().to_dict(), "| rate F13 %.2f F47 %.2f" % (R[A.farm == "F13"].mean(), R[A.farm == "F47"].mean()))
    print("  onset hour (half of rise reached):", A[R].onset.value_counts().sort_index().astype(int).to_dict())
    print("  level at 0h on rise days: median %.2f vs others %.2f; whole-day EC25 %.2f vs %.2f" % (A[R].lvl0.median(), A[~R].lvl0.median(), A[R].day.median(), A[~R].day.median()))
    # (a) calendar periodicity, pass 1
    for f in ("F13", "F47", "both"):
        Z = A[(~A.p2) & A.cal.notna() & ((A.farm == f) if f != "both" else True)].copy(); Z["d"] = np.round(Z.cal).astype(int)
        fl = (Z.up >= TH).values
        if fl.sum() < 6: continue
        ds = np.sort(np.unique(Z.d[fl])); gaps = np.diff(ds)
        res = []
        for P in range(2, 22):
            ph = (Z.d % P).values
            def st(fl_):
                r = np.array([fl_[ph == q].mean() if (ph == q).any() else 0 for q in range(P)]); return r.max() - fl_.mean()
            s0 = st(fl); nl = np.array([st(rng.permutation(fl)) for _ in range(1000)])
            res.append((P, s0, (nl >= s0).mean()))
        b = sorted(res, key=lambda x: x[2])[:4]
        print(f"  [{f}] rise dates (cal) gaps:", Counter(gaps.tolist()).most_common(8), "| best phase P:", ", ".join(f"P{P} +{s:.2f} p{p:.3f}" for P, s, p in b))
    # (b) same-date different-house synchrony
    pr = []
    for c, G in A.groupby("comp"):
        G = G[G.farm == G.farm.iloc[0]] if False else G
        for f, H in G.groupby("farm"):
            v = H.up.values
            for x in range(len(v)):
                for y in range(x + 1, len(v)): pr.append((v[x], v[y]))
    s = np.array(pr); rr = [np.corrcoef(*rng.choice(A.up.values, (2, len(s))))[0, 1] for _ in range(1000)]
    bo = ((s[:, 0] >= TH) & (s[:, 1] >= TH)).sum(); ei = ((s[:, 0] >= TH) | (s[:, 1] >= TH)).sum()
    print("  same date same farm other house: pairs %d, corr of up %.2f (null 95%% %.2f) | both-rise %d of %d pairs with any rise (indep. expect %.1f)" % (
        len(s), np.corrcoef(s[:, 0], s[:, 1])[0, 1], np.quantile(np.abs(rr), .95), bo, ei, ei * R.mean()))
    # (d) conditions on rise days vs same farm +-7 calendar days (season removed), pass 1 + pass 2 by number window
    rows = []
    for i in A.index[R]:
        f = A.at[i, "farm"]
        if A.at[i, "p2"]: nb = A[(A.farm == f) & A.p2 & ((A.num - A.at[i, "num"]).abs() <= 8) & ~R]
        else: nb = A[(A.farm == f) & ~A.p2 & ((A.cal - A.at[i, "cal"]).abs() <= 7) & ~R]
        if len(nb) >= 3: rows.append({k: A.at[i, k] - nb[k].median() for k in ["lvl0"] + list(IV)})
    Dd = pd.DataFrame(rows)
    if len(Dd):
        print("  rise day minus nearby non-rise days of same farm (median diff, share >0, n %d):" % len(Dd))
        print("   ", " | ".join(f"{k} {Dd[k].median():+.2f} ({(Dd[k] > 0).mean():.2f})" for k in Dd.columns))
# (e) hourly mean EC25 profile of rise days vs others (centered at 0..3h)
R = A.up >= .15
prof = lambda idx: np.nanmean(E25[idx] - np.nanmean(E25[idx, 0:4], 1, keepdims=True), 0)
print("\n[hourly EC25 change vs 0..3h, rise(+.15) days]:", np.round(prof(A.index[R].values), 2).tolist())
print("[same, other days]:", np.round(prof(A.index[~R].values), 2).tolist())
# hourly inputs on rise days vs others (z vs others, same farm)
print("\n[hourly input difference rise - other days, F13+F47 pooled, hours 0,4,8,10,12,14,16,20]")
for k in ["tin", "hum", "co2", "vent", "shade", "thermal", "heat", "fan", "st", "rad"]:
    v = IV[k]; a = np.nanmean(v[A.index[R].values], 0); b = np.nanmean(v[A.index[~R].values], 0)
    print(f"  {k:8s}", " ".join(f"{h}h {a[h]-b[h]:+.1f}" for h in (0, 4, 8, 10, 12, 14, 16, 20)))
# calendar listing of rise records, pass 1
print("\n[pass-1 rise(+.15) records by calendar position]")
Z = A[R & ~A.p2].sort_values("cal")
print("  ", ", ".join(f"{r.rec}@{r.cal:.0f}(+{r.up:.2f},lvl{r.lvl0:.2f})" for r in Z.itertuples()))
print("[pass-2 rise(+.15) records]:", ", ".join(f"{r.rec}(+{r.up:.2f},lvl{r.lvl0:.2f})" for r in A[R & A.p2].sort_values("num").itertuples()))
A.to_csv("rz2_records_v1.csv")
