# -*- coding: utf-8 -*-
"""HX9 / verify_episode V1 — 적대적 검증 (2026-10-13 집 클로드, 검증자 역할).
대상: EP1~EP3 '시간·에피소드 구조' 갈래의 signal/weak 판정.
계산 전에 적은 반박 가설 (지문):
 H-A 순환(자기 정답): truth a1 은 b 자신의 0시 EC(연장값)와 23시 EC 가 0.01 단위로 맞는 기록을 ~340개 학습 기록 중에서 고른 것.
     그렇다면 d_prev_h23 ~= ec_h0(b) 이고, r(Y1, d_prev_h23) 과 진단 오라클은 'b 자신의 0시 정답'으로 거의 같게 재현될 것.
     예측: r(Y1, ec_h0) >= .9, 오라클(자기 0시) RMSE 가 .148 과 비슷하거나 더 좋음, d_gap23 은 (ec_h0 - sg) 통제 후 잔차와 |부분 r| < .2.
 H-B 같은 날짜 후보 귀무: DM11 전날 날짜의 다른 학습 기록(동) 중 무작위 하나를 고르면 r 이 얼마나 떨어지나.
     입력(in_hum, in_co2, in_temp, 구동기)만으로 같은 날짜 후보 중 이음새가 가장 매끄러운 것을 고르면 a1 과 일치하나 (링크 정체를 입력이 독립 확인하는지).
 H-C 한 에피소드 의존: 에피소드 빼기 / hi12 기준에서 부호·크기 유지 여부.
 H-D u_pass2: 고EC 2차 5기록은 에피소드 2개(1, 4)에서만 → 에피소드 빼기로 무너질 것. 평가 60 은 전부 2차라 상수.
 H-E 놓친 변수: 같은 온실 번호 바로 앞 학습 기록(번호 인접)의 23시 EC.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings("ignore")
from feat4 import EC, ST, IN_, AC, FARM, NUM, TRAIN, N  # noqa
rng = np.random.default_rng(7)
RES = r"C:\work\farmai\집\클로드\research"
R = pd.read_csv(os.path.join(RES, "hx9", "data", "records.csv")).set_index("rec")
V = pd.read_csv(os.path.join(RES, "hx9", "episode", "vars_v2.csv")).set_index("rec")
HR = pd.read_csv(os.path.join(RES, "hx9", "data", "hourly.csv"))
D11 = pd.read_csv(os.path.join(RES, "ec0h", "rise", "dm11_calendar_v1.csv")).set_index("rec").day
print("hourly cols", list(HR.columns)[:20])

hi = R.ec_mean >= 1.0; hi12 = R.ec_mean >= 1.2; ok = R.has_sg == 1
Y1, Y2 = R.ec_mean, R.resid_mean
def cr(x, y, m):
    m = m & x.notna() & y.notna()
    if m.sum() < 5: return int(m.sum()), np.nan
    return int(m.sum()), stats.pearsonr(x[m], y[m])[0]
def pcorr(x, y, z, m):
    m = m & x.notna() & y.notna() & z.notna()
    A = np.column_stack([np.ones(m.sum()), z[m]])
    rx = x[m] - A @ np.linalg.lstsq(A, x[m], rcond=None)[0]; ry = y[m] - A @ np.linalg.lstsq(A, y[m], rcond=None)[0]
    return int(m.sum()), stats.pearsonr(rx, ry)[0]

print("\n=== H-A 자기 정답 순환 ===")
own_gap = R.ec_h0 - R.sg_mean
m = hi & ok & V.d_prev_h23.notna()
print(f"  |d_prev_h23 - ec_h0| 고EC 중앙 {np.median(np.abs(V.d_prev_h23[m]-R.ec_h0[m])):.3f}, 90% {np.quantile(np.abs(V.d_prev_h23[m]-R.ec_h0[m]),.9):.3f}; r(d_prev_h23, ec_h0) {cr(V.d_prev_h23, R.ec_h0, m)[1]:.3f}")
for nm, x in (("d_prev_h23", V.d_prev_h23), ("ec_h0 (자기)", R.ec_h0), ("d_prev_mean", V.d_prev_mean)):
    print(f"  Y1 ~ {nm:14s} 같은 행 n {cr(x, Y1, m)[0]} r {cr(x, Y1, m)[1]:.3f}")
for nm, x in (("d_gap23", V.d_gap23), ("ec_h0 - sg (자기)", own_gap), ("d_prev_amp", V.d_prev_amp), ("d_prev_mom", V.d_prev_mom), ("d_prev_resid", V.d_prev_resid), ("d_gap", V.d_gap)):
    n, r_ = cr(x, Y2, m); n2, pr = pcorr(x, Y2, own_gap, m)
    print(f"  Y2 ~ {nm:18s} n {n} r {r_:+.3f} | (자기 0시 갭) 통제 부분 r {pr:+.3f}")
# 오라클: 모델의 시각별 sg 모양을 유지하고 0시를 맞춤
H = [r for r in R.index[hi & ok] if isinstance(R.at[r, "prev_rec"], str)]
alt_prev = R.ec_mean[H] - (R.sg_mean[H] - R.sg_h0[H] + R.prev_ec_h23[H])
alt_own = R.ec_mean[H] - (R.sg_mean[H] - R.sg_h0[H] + R.ec_h0[H])
rm = lambda s: np.sqrt(np.mean(s ** 2))
print(f"  오라클 고EC n {len(H)}: 모델 {rm(Y2[H]):.3f} | 0시=진짜 전날 23시 {rm(alt_prev):.3f} | 0시=자기 0시 정답 {rm(alt_own):.3f}")

print("\n=== H-B 같은 날짜 후보 귀무 + 입력만으로 링크 정체 확인 ===")
idx = {f"{f}_{n:03d}": i for i, (f, n) in enumerate(zip(FARM, NUM))}
recs = list(R.index); d11 = D11.reindex(recs)
INV = np.concatenate([IN_, AC[..., :5]], 2)  # in_temp,in_hum,in_co2, vent,shade,thermal,heating,circfan
sdv = np.nanstd(np.diff(INV[TRAIN], axis=1).reshape(-1, INV.shape[2]), 0) + 1e-6
def seam_in(a, b, cols):
    pa = INV[a, 23, cols]; pb = INV[b, 0, cols]
    d = np.abs(pa - pb) / sdv[cols]
    return np.nanmean(d) if np.isfinite(d).any() else np.nan
COLSETS = {"in_hum+co2": [1, 2], "in_temp": [0], "act5": [3, 4, 5, 6, 7], "all8": list(range(8))}
rows = []
for set_nm, mask in (("hi10", hi), ("all", pd.Series(True, index=R.index))):
    for b in R.index[mask & R.prev_rec.notna()]:
        dd = d11.get(b)
        if not np.isfinite(dd): continue
        cands = [x for x in recs if x != b and TRAIN[idx[x]] and d11.get(x) == dd - 1]
        a = R.at[b, "prev_rec"]
        row = dict(set=set_nm, b=b, ncand=len(cands), a=a)
        for cn, cols in COLSETS.items():
            s = {x: seam_in(idx[x], idx[b], cols) for x in cands}
            s = {k: v for k, v in s.items() if np.isfinite(v)}
            if s:
                best = min(s, key=s.get); row[f"pick_{cn}"] = best; row[f"hit_{cn}"] = int(best == a)
                row[f"rank_{cn}"] = sorted(s.values()).index(s[a]) + 1 if a in s else np.nan
        rows.append(row)
B = pd.DataFrame(rows)
for set_nm in ("hi10", "all"):
    b_ = B[B.set == set_nm]
    print(f"  [{set_nm}] n {len(b_)} | 같은 날짜 후보 수 분포 {b_.ncand.value_counts().sort_index().to_dict()}")
    chance = np.mean(1 / b_.ncand[b_.ncand > 0])
    for cn in COLSETS:
        h_ = b_[f"hit_{cn}"].dropna(); rk = b_[f"rank_{cn}"].dropna()
        print(f"     입력 {cn:11s}: 입력 1순위 == a1 {h_.mean():.2f} (n {len(h_)}, 우연 {chance:.2f}) | a1 순위 평균 {rk.mean():.2f}")
bh = B[(B.set == "hi10") & (B.ncand >= 2)]
rs = []
CAND = {b: [x for x in recs if x != b and TRAIN[idx[x]] and d11.get(x) == d11.get(b) - 1] for b in bh.b}
for _ in range(2000):
    xs = [R.ec_h23[CAND[b][rng.integers(len(CAND[b]))]] for b in bh.b]
    rs.append(stats.pearsonr(xs, Y1[bh.b])[0])
xa = R.ec_h23[bh.a].values
print(f"  고EC n {len(bh)}: Y1 ~ a1 23시 r {stats.pearsonr(xa, Y1[bh.b])[0]:.3f} | 같은 날짜 무작위 후보 23시 r 평균 {np.mean(rs):.3f} [2.5% {np.quantile(rs,.025):.3f}, 97.5% {np.quantile(rs,.975):.3f}]")
for cn in ("in_hum+co2", "all8", "act5"):
    pk = bh[f"pick_{cn}"]; okk = pk.notna()
    print(f"  고EC: 입력 {cn} 로 고른 같은 날짜 후보 23시 EC 와 Y1 r {stats.pearsonr(R.ec_h23[pk[okk]].values, Y1[bh.b[okk]])[0]:.3f} (n {okk.sum()}) [날짜는 DM11 = 진단]")
B.to_csv(os.path.join(RES, "hx9", "verify_episode", "v1_cand_v1.csv"), index=False, encoding="utf-8-sig")

print("\n=== H-C 에피소드 빼기 / hi12 ===")
def loco_gain(x, y, sg, cl):
    m = x.notna() & y.notna() & sg.notna(); x, y, sg, cl = x[m].values, y[m].values, sg[m].values, cl[m].values
    if len(y) < 8: return np.nan
    pb = np.full(len(y), np.nan); pa = pb.copy()
    for c in np.unique(cl):
        te = cl == c; tr = ~te
        A = np.column_stack([np.ones(tr.sum()), sg[tr]]); w = np.linalg.lstsq(A, y[tr], rcond=None)[0]; pb[te] = w[0] + w[1] * sg[te]
        A = np.column_stack([np.ones(tr.sum()), sg[tr], x[tr]]); w = np.linalg.lstsq(A, y[tr], rcond=None)[0]; pa[te] = w[0] + w[1] * sg[te] + w[2] * x[te]
    sst = np.sum((y - y.mean()) ** 2)
    return (np.sum((y - pb) ** 2) - np.sum((y - pa) ** 2)) / sst
VV = V.copy(); VV["own_gap0"] = own_gap
CHK = ["d_prev_h23", "d_gap23", "d_gap", "d_prev_mean", "d_prev_resid", "d_prev_amp", "d_prev_mom", "d_trend1", "d_prev2_mean", "d_prev_max3", "d_succ_h0",
       "u_lk_steps", "u_lk_p1", "u_nd_n", "u_unclaimed_n", "u_tw_nhi", "u_dmt_seg", "u_pass2", "u_excl_mean", "own_gap0"]
EP = R.ep_id
out = []
for v in CHK:
    x = VV[v]; m = hi & ok
    n, r0 = cr(x, Y2, m); g0 = loco_gain(x[m], Y2[m], R.sg_mean[m], EP[m])
    n12, r12 = cr(x, Y2, hi12 & ok); g12 = loco_gain(x[hi12 & ok], Y2[hi12 & ok], R.sg_mean[hi12 & ok], EP[hi12 & ok])
    drs = []; dgs = []
    for e in sorted(EP[m].dropna().unique()):
        mm = m & (EP != e); drs.append(cr(x, Y2, mm)[1]); dgs.append(loco_gain(x[mm], Y2[mm], R.sg_mean[mm], EP[mm]))
    _, pr = pcorr(x, Y2, own_gap, m)
    out.append(dict(var=v, n=n, r=r0, gain=g0, r_drop_min=np.nanmin(drs), r_drop_max=np.nanmax(drs), gain_drop_min=np.nanmin(dgs),
                    n12=n12, r12=r12, gain12=g12, partial_own0=pr))
O = pd.DataFrame(out); pd.set_option("display.width", 250)
print(O.round(3).to_string(index=False))
O.to_csv(os.path.join(RES, "hx9", "verify_episode", "v1_dropep_v1.csv"), index=False, encoding="utf-8-sig")
# d_prev_amp 부분상관의 에피소드 빼기
m = hi & ok
prs = [pcorr(VV.d_prev_amp, Y2, own_gap, m & (EP != e))[1] for e in sorted(EP[m].dropna().unique())]
print(f"  d_prev_amp | 자기 0시 갭 통제 부분 r 에피소드 빼기 범위 [{min(prs):+.3f}, {max(prs):+.3f}]")
prs = [pcorr(VV.d_gap23, Y2, own_gap, m & (EP != e))[1] for e in sorted(EP[m].dropna().unique())]
print(f"  d_gap23 | 자기 0시 갭 통제 부분 r 에피소드 빼기 범위 [{min(prs):+.3f}, {max(prs):+.3f}]")

print("\n=== H-D u_pass2 ===")
for nm, mk in (("hi10", hi & ok), ("hi12", hi12 & ok), ("normal", (~hi) & ok)):
    for p in (0, 1):
        mm = mk & (R.pass2 == p)
        print(f"  {nm:6s} pass{p+1}: n {mm.sum():3d} resid mean {Y2[mm].mean():+.3f} | ec_h0-sg_h0 {(R.ec_h0-R.sg_h0)[mm].mean():+.3f}")
mm = hi & ok & (R.pass2 == 1)
print("  고EC 2차 기록:", list(zip(R.index[mm], EP[mm], Y2[mm].round(3))))
for e in EP[mm].unique():
    me = hi & ok & (EP == e)
    print(f"  ep {e}: pass1 resid {Y2[me & (R.pass2==0)].round(3).tolist()} | pass2 {Y2[me & (R.pass2==1)].round(3).tolist()}")
mk = hi & ok & EP.isin(EP[mm].unique())
yc = Y2[mk] - Y2[mk].groupby(EP[mk]).transform("mean"); xc = R.pass2[mk] - R.pass2[mk].groupby(EP[mk]).transform("mean")
sl = stats.linregress(xc, yc)
print(f"  에피소드 고정효과(1,4) 안 pass2 계수 {sl.slope:+.3f} p {sl.pvalue:.3f} n {mk.sum()}")
# sg 가 2차 고EC 에서 낮은 이유: 2차 고EC 의 sg vs 같은 에피소드 1차
for e in EP[mm].unique():
    me = hi & ok & (EP == e)
    print(f"  ep {e}: ec_mean p1 {Y1[me&(R.pass2==0)].mean():.3f} p2 {Y1[me&(R.pass2==1)].mean():.3f} | sg p1 {R.sg_mean[me&(R.pass2==0)].mean():.3f} p2 {R.sg_mean[me&(R.pass2==1)].mean():.3f}")

print("\n=== H-E 놓친 변수: 번호 인접 학습 기록 ===")
def lower_nb(b, same_farm=True):
    f, n = b[:3], int(b[4:])
    c = [x for x in recs if TRAIN[idx[x]] and x != b and int(x[4:]) < n and (x[:3] == f if same_farm else True)]
    return max(c, key=lambda x: int(x[4:])) if c else None
for sf in (True, False):
    nb = {b: lower_nb(b, sf) for b in recs}
    x = pd.Series({b: R.ec_h23[nb[b]] if nb[b] else np.nan for b in recs})
    gap = pd.Series({b: int(b[4:]) - int(nb[b][4:]) if nb[b] else np.nan for b in recs})
    hit = np.mean([nb[b] == R.at[b, "prev_rec"] for b in R.index[R.prev_rec.notna()]])
    hith = np.mean([nb[b] == R.at[b, "prev_rec"] for b in R.index[R.prev_rec.notna() & hi]])
    tag = "same-farm" if sf else "both-farms"
    print(f"  {tag} 번호 바로 앞 학습 기록: == 진짜 전날 {hit:.2f} (고EC {hith:.2f}) | 고EC Y1 r {cr(x, Y1, hi)[1]:+.3f} | 고EC Y2 ~ (x-sg) r {cr(x - R.sg_mean, Y2, hi & ok)[1]:+.3f} gain {loco_gain((x-R.sg_mean)[hi&ok], Y2[hi&ok], R.sg_mean[hi&ok], EP[hi&ok]):+.3f} | 전체 Y2 r {cr(x - R.sg_mean, Y2, ok)[1]:+.3f}")
    m2 = gap <= 2
    print(f"     간격<=2: 고EC n {(hi & ok & m2).sum()} Y2 r {cr(x - R.sg_mean, Y2, hi & ok & m2)[1]:+.3f}; 전체 n {(ok & m2).sum()} Y2 r {cr(x - R.sg_mean, Y2, ok & m2)[1]:+.3f}")
pr_ = R[R.prev_rec.notna()]
gg = [(int(b[4:]) - int(a[4:])) for b, a in zip(pr_.index, pr_.prev_rec) if a[:3] == b[:3]]
print("  진짜 전날(같은 온실) 번호 간격:", pd.Series(gg).value_counts().sort_index().head(12).to_dict())
gx = [(int(b[4:]) - int(a[4:])) for b, a in zip(pr_.index, pr_.prev_rec) if a[:3] != b[:3]]
print("  진짜 전날(다른 온실) 번호 간격:", pd.Series(gx).value_counts().sort_index().head(12).to_dict())
te = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"])
er = sorted({"_".join(s.split("_")[:2]) for s in te.row_id if s[:3] in ("F13", "F47")})
eg = []
for b in er:
    f, n = b[:3], int(b[4:])
    c = [int(x[4:]) for x in recs if x[:3] == f and int(x[4:]) < n and TRAIN[idx[x]]]
    eg.append(n - max(c) if c else np.nan)
print(f"  평가 {len(er)} 기록의 같은 온실 번호 앞 학습 기록 간격:", pd.Series(eg).value_counts().sort_index().to_dict())
