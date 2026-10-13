# -*- coding: utf-8 -*-
"""HX9 / verify_episode V2 — 놓친 변수 점검: '입력 이음새로 고른 전날 날짜 학습 기록'의 23시 EC (u_seam_ec23).
계산 전 예측: V1 에서 DM11 전날 날짜 후보 중 in_hum+in_co2 이음새 1순위가 a1 과 94% 일치(고EC). 날짜를 학습 전용 달력으로 바꿔도
(대상 자신의 0시 실외 쌍둥이 -> 쌍둥이의 DMT 날짜 -1) 고EC 에서 r(Y1) >= .7 이 남으면, EP1 의 '사용판 실패'는 링커 CV 설정(같은 온실 후보 + 블록 가림) 탓이지
지속 정보 자체가 사용 불가능해서가 아님. 남지 않으면 EP1 결론 유지.
사용 가능성: b 의 0시 입력(같은 기록 h=0), 학습 기록 입력·정답만 사용 -> h=0 부터 계산 가능 (가림: b 자신 제외).
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings("ignore")
from feat4 import EC, IN_, AC, X, FARM, NUM, TRAIN  # noqa
RES = r"C:\work\farmai\집\클로드\research"
R = pd.read_csv(os.path.join(RES, "hx9", "data", "records.csv")).set_index("rec")
DMT = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")).set_index("rec")
idx = {f"{f}_{n:03d}": i for i, (f, n) in enumerate(zip(FARM, NUM))}
recs = list(R.index)
INV = np.concatenate([IN_, AC[..., :5]], 2)
sdv = np.nanstd(np.diff(INV[TRAIN], axis=1).reshape(-1, INV.shape[2]), 0) + 1e-6
hi = R.ec_mean >= 1.0; hi12 = R.ec_mean >= 1.2; ok = R.has_sg == 1
Y1, Y2 = R.ec_mean, R.resid_mean; EP = R.ep_id
# 0시 실외 쌍둥이 (학습 기록, b 제외)
key0 = {r: tuple(np.round(X[idx[r], 0], 1)) for r in idx}
by0 = {}
for r, k in key0.items():
    if TRAIN[idx[r]]: by0.setdefault(k, []).append(r)
DMTd = {r: (int(DMT.at[r, "seg"]), int(DMT.at[r, "day"])) for r in DMT.index}
bydate = {}
for r, sd in DMTd.items():
    if r in idx and TRAIN[idx[r]]: bydate.setdefault(sd, []).append(r)
def seam(a, b, cols=(1, 2)):
    cols = list(cols); d = np.abs(INV[a, 23, cols] - INV[b, 0, cols]) / sdv[cols]
    return np.nanmean(d) if np.isfinite(d).any() else np.inf
rows = []
for b in recs:
    tw = [x for x in by0.get(key0[b], []) if x != b and x in DMTd]
    dates = {DMTd[x] for x in tw}
    row = dict(rec=b, ntw=len(tw), ndates=len(dates))
    if len(dates) == 1:
        s, d = dates.pop()
        cands = [x for x in bydate.get((s, d - 1), []) if x != b]
        row["ncand"] = len(cands)
        if cands:
            sc = {x: seam(idx[x], idx[b]) for x in cands}
            best = min(sc, key=sc.get); o = sorted(sc.values())
            row.update(pick=best, seam1=o[0], margin=(o[1] - o[0]) if len(o) > 1 else np.nan,
                       u_seam_ec23=R.ec_h23[best], u_seam_mean=R.ec_mean[best], hit=int(best == R.at[b, "prev_rec"]) if isinstance(R.at[b, "prev_rec"], str) else np.nan)
    rows.append(row)
S = pd.DataFrame(rows).set_index("rec")
S["u_seam_gap"] = S.u_seam_ec23 - R.sg_mean
S.to_csv(os.path.join(RES, "hx9", "verify_episode", "v2_seam_pick_v1.csv"), encoding="utf-8-sig")
def cr(x, y, m):
    m = m & x.notna() & y.notna(); return int(m.sum()), (stats.pearsonr(x[m], y[m])[0] if m.sum() > 4 else np.nan)
def loco_gain(x, y, sg, cl):
    m = x.notna() & y.notna() & sg.notna(); x, y, sg, cl = x[m].values, y[m].values, sg[m].values, cl[m].values
    pb = np.full(len(y), np.nan); pa = pb.copy()
    for c in np.unique(cl):
        te = cl == c; tr = ~te
        A = np.column_stack([np.ones(tr.sum()), sg[tr]]); w = np.linalg.lstsq(A, y[tr], rcond=None)[0]; pb[te] = w[0] + w[1] * sg[te]
        A = np.column_stack([np.ones(tr.sum()), sg[tr], x[tr]]); w = np.linalg.lstsq(A, y[tr], rcond=None)[0]; pa[te] = w[0] + w[1] * sg[te] + w[2] * x[te]
    return (np.sum((y - pb) ** 2) - np.sum((y - pa) ** 2)) / np.sum((y - y.mean()) ** 2), int(m.sum())
print("coverage: date unique", int((S.ndates == 1).sum()), "/400 | pick", int(S.pick.notna().sum()))
for nm, m in (("hi10", hi), ("hi12", hi12), ("all", pd.Series(True, index=R.index))):
    mm = m & S.pick.notna()
    print(f"[{nm}] pick n {mm.sum()}/{m.sum()} | hit vs truth prev {S.hit[mm].mean():.2f} (n {S.hit[mm].notna().sum()}) | Y1 r {cr(S.u_seam_ec23, Y1, m)[1]:.3f}")
    mo = m & ok
    g, n = loco_gain(S.u_seam_gap[mo], Y2[mo], R.sg_mean[mo], EP[mo] if nm != "all" else pd.Series(np.arange(len(R)), index=R.index)[mo])
    print(f"      Y2 ~ u_seam_gap r {cr(S.u_seam_gap, Y2, mo)[1]:+.3f} | LOCO(에피소드/기록) R2 이득 {g:+.3f} n {n}")
    # 고EC 에피소드 빼기
    if nm == "hi10":
        drs = [cr(S.u_seam_gap, Y2, mo & (EP != e))[1] for e in sorted(EP[mo].dropna().unique())]
        print(f"      에피소드 빼기 r 범위 [{min(drs):+.3f}, {max(drs):+.3f}]")
        # 확신 구간 (margin 큰 것)
        for th in (0.5, 1.0, 2.0):
            mt = mo & (S.margin >= th)
            print(f"      margin>={th}: n {mt.sum()} hit {S.hit[mt].mean():.2f} Y2 r {cr(S.u_seam_gap, Y2, mt)[1]:+.3f}")
# 오라클형 대체: 0시를 u_seam_ec23 으로 (모양은 모델)
for nm, m in (("hi10", hi & ok), ("normal", (~hi) & ok)):
    mm = m & S.pick.notna()
    alt = Y1[mm] - (R.sg_mean[mm] - R.sg_h0[mm] + S.u_seam_ec23[mm])
    print(f"  [{nm}] n {mm.sum()}: RMSE 모델 {np.sqrt(np.mean(Y2[mm]**2)):.3f} -> 0시=입력 이음새 픽 23시 {np.sqrt(np.mean(alt**2)):.3f}; 개선 {int((alt.abs()<Y2[mm].abs()).sum())}/{mm.sum()}")
# 1차/2차
for p in (0, 1):
    mm = ok & S.pick.notna() & (R.pass2 == p)
    alt = Y1[mm] - (R.sg_mean[mm] - R.sg_h0[mm] + S.u_seam_ec23[mm])
    print(f"  pass{p+1} 전체 n {mm.sum()}: RMSE {np.sqrt(np.mean(Y2[mm]**2)):.3f} -> {np.sqrt(np.mean(alt**2)):.3f}; hit {S.hit[mm].mean():.2f}")
# 평가 60 포괄
te = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"])
er = sorted({"_".join(s.split("_")[:2]) for s in te.row_id if s[:3] in ("F13", "F47")})
cov = []
for b in er:
    tw = [x for x in by0.get(key0[b], []) if x in DMTd]; dates = {DMTd[x] for x in tw}
    c = dict(rec=b, ntw=len(tw), ndates=len(dates))
    if len(dates) == 1:
        s, d = dates.pop(); cands = bydate.get((s, d - 1), [])
        if cands:
            sc = {x: seam(idx[x], idx[b]) for x in cands}; best = min(sc, key=sc.get); o = sorted(sc.values())
            c.update(ncand=len(cands), pick=best, seam1=o[0], margin=(o[1] - o[0]) if len(o) > 1 else np.nan, ec23=R.ec_h23[best])
    cov.append(c)
C = pd.DataFrame(cov)
print(f"평가 60: 날짜 1개 {int((C.ndates==1).sum())} | 픽 있음 {int(C.pick.notna().sum())} | 픽 23시 >=1.0 {int((C.ec23>=1.0).sum())} | 학습 seam1 분포(고EC 히트) 참고")
print("  학습 hit 기록 seam1 중앙", round(S.seam1[S.hit == 1].median(), 3), "miss 중앙", round(S.seam1[S.hit == 0].median(), 3), "| 평가 seam1 중앙", round(C.seam1.median(), 3))
print(C[C.ec23 >= 1.0].round(3).to_string(index=False))
C.to_csv(os.path.join(RES, "hx9", "verify_episode", "v2_eval_cov_v1.csv"), index=False, encoding="utf-8-sig")
