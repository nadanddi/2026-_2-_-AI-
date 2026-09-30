# -*- coding: utf-8 -*-
"""재분석 2단계 탐색 Q1·Q3 (2026-10-01, 집 클로드). 모델 선택 없음, 진단만.

Q1: G_C2 OOF 오차 구조 (하루 오프셋 vs 하루 안 모양, 시간대·온도구간·온실별)
Q3: 하루 안 잔차가 실내온도 동역학(변화율·지연)과 관련 있는가

G_C2 재구성 (연구실/클로드/code/gc2_error_anatomy_v1.py와 같은 식):
  g = clip((in_temp-8)/2,0,1); pred=(0.6-0.2(1-g))MASK + (0.2+0.4(1-g))CODEX + 0.2g TabPFN
결과: local/re02_q1q3.txt
"""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
from harness import load

OUT = os.path.join(env.LOCAL, "re02_q1q3.txt")
_f = open(OUT, "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _f.write(s + "\n")


def rmse(e):
    return float(np.sqrt(np.nanmean(np.asarray(e) ** 2)))


def gc2(lab, z, split):
    base = np.nanmean([z[f"{split}__MASK__7"], z[f"{split}__MASK__101"]], axis=0)
    cx = np.nanmean([z[f"{split}__CODEX__726"], z[f"{split}__CODEX__727"]], axis=0)
    pfn = np.load(os.path.join(env.LOCAL, f"web_tabpfn_v2_temp_{split}.npy")).mean(0)
    t = lab.in_temp.values
    g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    return (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn


def boot_ci_corr(x, y, groups, n=2000, seed=0):
    """날 단위 블록 부트스트랩 상관 95% CI."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {gg: np.where(groups == gg)[0] for gg in ug}
    rs = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug))
        ii = np.concatenate([idx[gg] for gg in pick])
        rs.append(np.corrcoef(x[ii], y[ii])[0, 1])
    return np.percentile(rs, [2.5, 97.5])


_, lab, _ = load()
z = np.load(os.path.join(env.LOCAL, "temp_mask_v1_oof.npz"), allow_pickle=True)
assert (lab.row_id.values == z["row_id"]).all()
lab = lab.sort_values(["farm", "day", "hour"]).reset_index()  # 'index' = 원래 위치
order = lab["index"].values

for split in ("EXT10", "DIAG10"):
    pred = gc2(lab.set_index("index").loc[order].reset_index(), z, split)[order] if False else None
    base_pred = gc2(load()[1], z, split)[order]
    d = lab.copy()
    d["pred"] = base_pred
    d = d[~np.isnan(d.pred)].copy()
    d["e"] = d.pred - d.sub_temp
    d["dayid"] = d.farm + "_" + d.day.astype(str)
    p("=" * 100)
    p(f"[{split}] 행 {len(d)} 날 {d.dayid.nunique()} G_C2 RMSE {rmse(d.e):.4f} 편향 {d.e.mean():+.4f}")

    # ---- Q1 구조
    d["e_day"] = d.groupby("dayid").e.transform("mean")
    d["e_in"] = d.e - d.e_day
    sse = (d.e ** 2).sum()
    p(f"  SSE 비중: 하루 오프셋 {100*(d.e_day**2).sum()/sse:.1f}% | 하루 안 {100*(d.e_in**2).sum()/sse:.1f}%")
    dd = d.groupby("dayid").e.mean()
    p(f"  하루 오프셋 분포: sd {dd.std():.3f}, |오프셋|>0.5 날 {int((dd.abs()>0.5).sum())}/{len(dd)}, "
      f"상위 10% 날이 오프셋 SSE의 {100*np.sort(dd.values**2)[::-1][:max(1,len(dd)//10)].sum()/(dd**2).sum():.1f}%")
    p("  시간대별 RMSE / 편향 / 하루안 RMSE:")
    hh = d.groupby(d.hour // 3 * 3).agg(rmse=("e", rmse), bias=("e", "mean"), rin=("e_in", rmse), n=("e", "size"))
    p(hh.round(3).to_string())
    band = pd.cut(d.in_temp, [-99, 8, 10, 12, 15, 20, 99])
    bb = d.groupby(band, observed=True).agg(rmse=("e", rmse), bias=("e", "mean"), n=("e", "size"),
                                             sse_pct=("e", lambda s: 100 * (s ** 2).sum() / sse))
    p("  실내온도 구간별:")
    p(bb.round(3).to_string())
    ff = d.groupby("farm").agg(rmse=("e", rmse), bias=("e", "mean"), day_sd=("e_day", lambda s: s.groupby(d.loc[s.index,'dayid']).first().std()))
    p("  온실별:")
    p(ff.round(3).to_string())

    # ---- Q3 하루 안 잔차 vs 동역학 (같은 날 안에서만 차분/지연; 인과)
    g = d.groupby("dayid")
    d["dT1"] = g.in_temp.diff()
    d["dT3"] = d.in_temp - g.in_temp.shift(3)
    d["lag1"] = g.in_temp.shift(1)
    d["accel"] = g.dT1.diff()
    d["drad1"] = g.out_rad.diff()
    d["dheat1"] = g.act_heating.diff()
    d["dvent1"] = g.act_vent.diff()
    d["dthermal1"] = g.act_thermal.diff()
    # 라벨 쪽 동역학(설명용, 특징 아님): 배지 변화율
    d["dsub1"] = g.sub_temp.diff()
    cands = ["dT1", "dT3", "accel", "drad1", "dheat1", "dvent1", "dthermal1"]
    p(f"  Q3: 하루 안 잔차 e_in과 인과 동역학 후보 상관 (검정 {len(cands)}개, 날 블록 부트스트랩 95% CI)")
    for c in cands:
        m = d[[c, "e_in"]].dropna()
        r = np.corrcoef(m[c], m.e_in)[0, 1]
        lo, hi = boot_ci_corr(m[c].values, m.e_in.values, d.loc[m.index, "dayid"].values, n=500)
        p(f"    {c:10s} r {r:+.3f}  CI [{lo:+.3f},{hi:+.3f}]  n {len(m)}")
    # 층화: 온실별, 낮/밤
    p("  Q3 층화 (dT1, 온실×낮밤):")
    for (fm, dn), grp in d.assign(dn=np.where(d.hour.between(8, 17), "낮", "밤")).groupby(["farm", "dn"]):
        m = grp[["dT1", "e_in"]].dropna()
        p(f"    {fm} {dn}: r {np.corrcoef(m.dT1, m.e_in)[0,1]:+.3f} n {len(m)}")
    # 배지 반응 지연: 하루 안에서 dsub(h) vs dT(h-k) 상관
    p("  배지 반응 지연 (같은 날 안 dsub_temp(h) vs dT_in(h-k), 설명용):")
    for k in range(0, 5):
        x = g.dT1.shift(k)
        m = pd.concat([d.dsub1, x], axis=1).dropna()
        p(f"    k={k}: r {np.corrcoef(m.iloc[:,0], m.iloc[:,1])[0,1]:+.3f}")
    # 예측의 변화율이 실제 변화율을 얼마나 따라가나
    d["dpred1"] = g.pred.diff()
    m = d[["dpred1", "dsub1"]].dropna()
    slope = np.polyfit(m.dpred1, m.dsub1, 1)[0]
    p(f"  예측 변화율 vs 실제 변화율: r {np.corrcoef(m.dpred1, m.dsub1)[0,1]:.3f}, 기울기(실제~예측) {slope:.3f}, "
      f"sd 예측 {m.dpred1.std():.3f} 실제 {m.dsub1.std():.3f}")

_f.close()
