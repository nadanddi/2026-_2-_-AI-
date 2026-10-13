# -*- coding: utf-8 -*-
"""HX9 verify_history v1 (2026-10-13 집 클로드, adversarial verifier).
대상: hx9/history 의 weak 판정 변수 (vpd_p3, et0_p3, trans_p1, lowrad_run_dm11, ttrend5_dm11) 와 다변량 상한.
반박 시도 (계산 전에 고정):
  V1 이중계산: vpd_p3 의 잔차 상관이 '수준(y)' 이 아니라 '모델 예측(sg)' 쪽에서 오는가? 당일 입력(당일 외기 VPD, 실내 습도·온도)
     통제 후에도 남는가? (모델은 당일 입력을 이미 봄 → 당일 날씨와 자기상관된 3일 평균은 모델 반응의 대리일 수 있음)
  V2 유효 표본: 날짜 수준 변수 + 에피소드 연속일 → 에피소드 평균끼리 상관(n 8)과 에피소드 안(평균 제거) 상관으로 분해.
  V3 표본 선택: p3 가 결측인 고EC (에피소드 시작/구간 첫날) 를 빼서 생긴 상관인가?
  V4 trans_p1: Y3 = EC(b)-EC(전날) 인데 trans_p1 은 전날 기록의 입력 → 전날 EC 와의 상관(평균 회귀)으로 생긴 것인가? 농장·에피소드 교란?
  V5 다중검정: 56 변수 x 4 목표에서 명목 p<.05 기대 개수 vs 관찰 개수.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE)
R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
H = pd.read_csv(os.path.join(HX9, "data", "hourly.csv"))
V = pd.read_csv(os.path.join(HX9, "history", "vars_v1.csv")).set_index("rec")
S = pd.read_csv(os.path.join(HX9, "history", "stats_v1.csv"))
RNG = np.random.default_rng(7)

def es(T): return 0.6108 * np.exp(17.27 * T / (T + 237.3))
H["ovpd"] = es(H.out_temp) * (1 - H.out_hum / 100); H["ivpd"] = es(H.in_temp) * (1 - H.in_hum / 100)
day = H[(H.hour >= 9) & (H.hour <= 16)].groupby("rec")
cur = pd.DataFrame({"vpd_d0": day.ovpd.mean(), "invpd_d0": day.ivpd.mean(),
                    "inhum_d0": H.groupby("rec").in_hum.mean(), "intemp_d0": H.groupby("rec").in_temp.mean(),
                    "shade_d0": H.groupby("rec").act_shade.mean(),
                    "vpd_0to8": H[H.hour <= 8].groupby("rec").ovpd.mean()})
D = R.join(V, rsuffix="_v").join(cur)
D["Y3"] = D.ec_mean - D.prev_ec_mean
hiS = D[(D.hi10 == 1) & (D.has_sg == 1)].copy()
hi3 = D[(D.hi10 == 1) & D.prev_ec_mean.notna()].copy()

def r_(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4: return np.nan, np.nan, int(m.sum())
    r, p = stats.pearsonr(a[m], b[m]); return r, p, int(m.sum())

def partial(df, x, y, ctl):
    sub = df[[x, y] + ctl].dropna()
    A = np.c_[np.ones(len(sub)), sub[ctl].values]
    rx = sub[x].values - A @ np.linalg.lstsq(A, sub[x].values, rcond=None)[0]
    ry = sub[y].values - A @ np.linalg.lstsq(A, sub[y].values, rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1], len(sub)

def decompose(df, x, y, g="ep_id"):
    sub = df[[x, y, g]].dropna()
    w = sub.groupby(g)[[x, y]].transform(lambda s: s - s.mean())
    keep = sub.groupby(g)[x].transform("size") >= 2
    rw = np.corrcoef(w[x][keep], w[y][keep])[0, 1] if keep.sum() > 4 else np.nan
    mb = sub.groupby(g)[[x, y]].mean()
    rb, pb = stats.pearsonr(mb[x], mb[y]) if len(mb) >= 4 else (np.nan, np.nan)
    # exact-ish permutation of episode means
    perm = [abs(np.corrcoef(RNG.permutation(mb[x].values), mb[y].values)[0, 1]) for _ in range(5000)]
    return rw, int(keep.sum()), rb, pb, len(mb), np.mean(np.array(perm) >= abs(rb))

print("=== V1 double counting with model prediction / same-day inputs (hi10 with OOF) ===")
for x in ["vpd_p3", "et0_p3", "vpd_p3_dm11", "trans_p1", "lowrad_run_dm11"]:
    a = r_(hiS[x].values, hiS.resid_mean.values); b = r_(hiS[x].values, hiS.ec_mean.values); c = r_(hiS[x].values, hiS.sg_mean.values)
    print(f"{x:16s} r(x,resid) {a[0]:+.3f} (n {a[2]}) | r(x,y) {b[0]:+.3f} | r(x,sg) {c[0]:+.3f}")
    for ctl in (["vpd_d0"], ["vpd_0to8"], ["vpd_d0", "inhum_d0", "intemp_d0"], ["sg_mean"], ["invpd_d0"]):
        pr, n = partial(hiS, x, "resid_mean", ctl); py, _ = partial(hiS, x, "ec_mean", ctl + (["sg_mean"] if "sg_mean" not in ctl else []))
        print(f"      | ctl {'+'.join(ctl):28s} partial r(x,resid) {pr:+.3f}  partial r(x,y|ctl+sg) {py:+.3f} (n {n})")
print("current-day comparators vs resid among hi10:")
for x in ["vpd_d0", "vpd_0to8", "invpd_d0", "inhum_d0", "intemp_d0", "shade_d0"]:
    a = r_(hiS[x].values, hiS.resid_mean.values); c = r_(hiS[x].values, hiS.vpd_p3.values)
    print(f"   {x:10s} r(x,resid) {a[0]:+.3f} p {a[1]:.3f} (n {a[2]}) | r(x, vpd_p3) {c[0]:+.3f}")

print("\n=== V2 between-episode vs within-episode decomposition ===")
for x, y, df in (("vpd_p3", "resid_mean", hiS), ("et0_p3", "resid_mean", hiS), ("vpd_p3_dm11", "resid_mean", hiS),
                 ("trans_p1", "resid_mean", hiS), ("trans_p1", "Y3", hi3), ("ttrend5_dm11", "Y3", hi3), ("lowrad_run_dm11", "resid_mean", hiS)):
    rw, nw, rb, pb, nb, pp = decompose(df, x, y)
    print(f"{x:16s} vs {y:10s}: within-episode r {rw:+.3f} (n {nw}) | episode-mean r {rb:+.3f} (n {nb}, p {pb:.3f}, perm p {pp:.3f})")
sub = hiS[["vpd_p3", "dmt_seg", "dmt_day"]].dropna()
print(f"vpd_p3 among hi10: n {len(sub)}, distinct dates (seg,day) {sub.groupby(['dmt_seg','dmt_day']).ngroups}, distinct values {sub.vpd_p3.round(6).nunique()}")

print("\n=== V3 selection: hi10 records with missing vpd_p3 ===")
miss = hiS[hiS.vpd_p3.isna()]
print(miss[["pass2", "dmt_seg", "dmt_day", "ep_id", "resid_mean", "vpd_p3_dm11"]].round(3).to_string())
print(f"mean resid: with p3 {hiS[hiS.vpd_p3.notna()].resid_mean.mean():+.3f} | missing p3 {miss.resid_mean.mean():+.3f}")
# if the relation were real, the DM11 copy for the missing records should predict their resid via the fit on the others
ok = hiS.dropna(subset=["vpd_p3_dm11"])
fit = np.polyfit(ok.loc[hiS.vpd_p3.notna(), "vpd_p3_dm11"].dropna(), ok.loc[hiS.vpd_p3.notna()].dropna(subset=["vpd_p3_dm11"]).resid_mean, 1)
mm = miss.dropna(subset=["vpd_p3_dm11"])
print(f"DM11-copy fit on p3-available records applied to missing ones: pred {np.polyval(fit, mm.vpd_p3_dm11).round(2).tolist()} vs actual {mm.resid_mean.round(2).tolist()}")
for x in ["vpd_p3_dm11", "et0_p3_dm11"]:
    a = r_(hiS[x].values, hiS.resid_mean.values); b = r_(hiS.loc[hiS.vpd_p3.notna(), x].values, hiS.loc[hiS.vpd_p3.notna()].resid_mean.values)
    print(f"   {x}: r over all 31 {a[0]:+.3f} | r over the 26 with DMT3e p3 {b[0]:+.3f}")

print("\n=== V4 trans_p1 vs Y3: regression-to-mean / house confounding ===")
for y in ["Y3", "ec_mean", "prev_ec_mean"]:
    a = r_(hi3.trans_p1.values, hi3[y].values); print(f"   r(trans_p1, {y}) {a[0]:+.3f} p {a[1]:.3f} n {a[2]}")
pr, n = partial(hi3, "trans_p1", "ec_mean", ["prev_ec_mean"]); print(f"   partial r(trans_p1, EC(b) | EC(prev)) {pr:+.3f} n {n}")
D["prev_shade"] = [D.at[p, "shade_d0"] if isinstance(p, str) and p in D.index else np.nan for p in D.prev_rec]
D["prev_rad"] = [D.at[p, "out_rad_d"] if isinstance(p, str) and p in D.index else np.nan for p in D.prev_rec]
hi3 = hi3.join(D[["prev_shade", "prev_rad"]])
for x in ["prev_shade", "prev_rad", "shade_d0"]:
    a = r_(hi3[x].values, hi3.Y3.values); print(f"   r({x}, Y3) {a[0]:+.3f} p {a[1]:.3f} n {a[2]}")
for f in ["F13", "F47"]:
    s = hi3[hi3.farm == f]; a = r_(s.trans_p1.values, s.Y3.values); print(f"   within {f}: r(trans_p1,Y3) {a[0]:+.3f} n {a[2]}")
s = hi3[hi3.prev_same_farm == 1]; a = r_(s.trans_p1.values, s.Y3.values); print(f"   prev link same farm only: r {a[0]:+.3f} n {a[2]}")
# usable surrogate: own-record shade/light at hours 0..h is not prev-day; same-ID previous record number (num-?) is not same house. Report own-day light sum as the closest usable proxy
H["light"] = H.out_rad * 0.0036 * (1 - H.act_shade.fillna(0) / 100)
D["light_d0"] = H.groupby("rec").light.sum(); hi3 = hi3.join(D[["light_d0"]])
a = r_(hi3.light_d0.values, hi3.Y3.values); print(f"   own-day light sum (U23) r with Y3 {a[0]:+.3f} n {a[2]}")
# bootstrap over episodes of Y3 r with trans_p1, and leave-one-episode-out
sub = hi3[["trans_p1", "Y3", "ep_id"]].dropna(); gs = sub.ep_id.unique(); out = []
for _ in range(4000):
    pick = RNG.choice(gs, len(gs)); ii = pd.concat([sub[sub.ep_id == g] for g in pick])
    if ii.trans_p1.std() > 0: out.append(np.corrcoef(ii.trans_p1, ii.Y3)[0, 1])
print(f"   episode bootstrap r(trans_p1,Y3) 95% [{np.percentile(out,2.5):+.2f},{np.percentile(out,97.5):+.2f}]  (teammate reported [+.11,+.52])")

print("\n=== V5 multiple testing over the reported table ===")
pcols = {}
for t, rc, nc in (("Y1", "r_Y1", "n_Y1"), ("Y2h", "r_Y2h", "n_Y2h"), ("Y2a", "r_Y2a", "n_Y2a"), ("Y3", "r_Y3", "n_Y3")):
    r = S[rc].values; n = S[nc].values
    t_ = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-9, None)); p = 2 * stats.t.sf(np.abs(t_), n - 2)
    pcols[t] = p
P = np.concatenate([v[np.isfinite(v)] for v in pcols.values()])
print(f"   tests {len(P)} | nominal p<.05 observed {np.sum(P<.05)} expected {0.05*len(P):.1f} | p<.01 observed {np.sum(P<.01)} expected {0.01*len(P):.1f} | min p {P.min():.4f} | Bonferroni .05/{len(P)} = {0.05/len(P):.5f}")
Ps = np.sort(P); bh = np.where(Ps <= 0.10 * np.arange(1, len(Ps) + 1) / len(Ps))[0]
print(f"   BH q<.10 survivors: {0 if len(bh)==0 else bh.max()+1}")
for t, p in pcols.items():
    idx = np.argsort(np.nan_to_num(p, nan=1))[:3]
    print(f"   {t}: smallest p " + ", ".join(f"{S.variable[i]} {p[i]:.3f}" for i in idx))

print("\n=== V6 within-episode time trend (ep_pos) as confounder of vpd_p3/et0_p3 ===")
for x in ["vpd_p3", "et0_p3"]:
    sub = hiS[[x, "resid_mean", "ep_id", "ep_pos"]].dropna()
    w = sub.groupby("ep_id")[[x, "resid_mean", "ep_pos"]].transform(lambda s: s - s.mean())
    print(f"   {x}: within-episode r(x,ep_pos) {np.corrcoef(w[x], w.ep_pos)[0,1]:+.3f} | r(resid,ep_pos) {np.corrcoef(w.resid_mean, w.ep_pos)[0,1]:+.3f} | "
          f"within r(x,resid | ep_pos) {partial(w.assign(x_=w[x]), 'x_', 'resid_mean', ['ep_pos'])[0]:+.3f}")
    # leave-one-episode-out slope prediction from within-episode slope (demeaned): does within slope transfer?
    sse0 = sse1 = 0
    for g in sub.ep_id.unique():
        tr, te = w[sub.ep_id != g], w[sub.ep_id == g]
        if len(te) < 2: continue
        b = np.polyfit(tr[x], tr.resid_mean, 1)[0]
        sse0 += np.sum(te.resid_mean ** 2); sse1 += np.sum((te.resid_mean - b * te[x]) ** 2)
    print(f"      within-episode LOEO: SSE no-x {sse0:.4f} -> with slope {sse1:.4f} (gain {1-sse1/sse0:+.3f}) [needs episode mean known = diagnostic]")
