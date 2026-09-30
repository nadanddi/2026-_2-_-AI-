# -*- coding: utf-8 -*-
"""재분석 17 (2026-10-01, 집 클로드): 하루 오차 큰 날·작은 날을 "처리구" 관점으로 해부 — 탐색(기술), 모델 없음.

배경: 원천은 연구용 실험 온실(사용자). 한 온실 ID 안 같은 날짜 두 기록 = 같은 날 다른 처리구라는 해석(6b.31).
기존: 같은 날짜 다른 출처 오프셋 상관 0.20(6b.3), 홀짝 위상 특징 기각(9절).
A. G_C2 DIAG10 하루 오프셋 상위 과대 10·과소 10·최소 20일의 하루 프로필 비교 (라벨 기반 서술 포함)
B. 같은 온실 형제일 짝(외기 24시간 완전 일치): 오프셋 차이 vs 입력 차이 상관 — 날씨·날짜를 지운 대조
C. 오차 최대 3일과 형제일 시간별 나란히 출력
결과 local/re17_day_dissection.txt
"""
import env  # noqa
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
from harness import load

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
out = open(env.LOCAL + "/re17_day_dissection.txt", "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")

_, lab, _ = load()
z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
t = lab.in_temp.values; g = np.where(np.isnan(t), 1, np.clip((t - 8) / 2, 0, 1)); s = "DIAG10"
base = np.nanmean([z[f"{s}__MASK__7"], z[f"{s}__MASK__101"]], 0)
cx = np.nanmean([z[f"{s}__CODEX__726"], z[f"{s}__CODEX__727"]], 0)
pfn = np.load(env.LOCAL + f"/web_tabpfn_v2_temp_{s}.npy").mean(0)
lab = lab.assign(pred=(0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn)
lab["e"] = lab.pred - lab.sub_temp
lab["sub_in"] = lab.sub_temp - lab.in_temp
lab = lab.sort_values(["farm", "day", "hour"]).reset_index(drop=True)
night = lab.hour.between(0, 6)

agg = dict(e_day=("e", "mean"), tin=("in_temp", "mean"), tin_min=("in_temp", "min"), tin_max=("in_temp", "max"),
           tout=("out_temp", "mean"), rad=("out_rad", "sum"), hum=("in_hum", "mean"), co2=("in_co2", "mean"),
           heat=("act_heating", "mean"), therm=("act_thermal", "mean"), vent=("act_vent", "mean"),
           shade=("act_shade", "mean"), circ=("act_circfan", "mean"), co2a=("act_co2", "mean"), fog=("act_fog", "mean"),
           sub=("sub_temp", "mean"), sub_in=("sub_in", "mean"), sub_amp=("sub_temp", lambda q: q.max() - q.min()))
D = lab.groupby(["farm", "day"]).agg(**agg).reset_index()
Nn = lab[night].groupby(["farm", "day"]).agg(heat_n=("act_heating", "mean"), therm_n=("act_thermal", "mean"),
                                             tin_n=("in_temp", "mean"), sub_in_n=("sub_in", "mean")).reset_index()
D = D.merge(Nn, on=["farm", "day"])
D["in_amp"] = D.tin_max - D.tin_min
D["pass"] = np.where(D.day >= 179, 2, 1)

# 형제일: 외기 4변수 24시간 완전 일치 (학습 F13·F47)
W = lab.groupby(["farm", "day"]).apply(lambda q: tuple(np.round(q[["out_temp", "out_hum", "out_rad", "out_wspd"]].values.ravel(), 1)) if len(q) == 24 else None)
W = W.dropna().rename("sig").reset_index()
grp = W.groupby("sig").apply(lambda q: list(zip(q.farm, q.day)))
sib = {}
for members in grp:
    for m in members:
        sib[m] = [x for x in members if x != m]
D["sibs"] = [sib.get((f, d), []) for f, d in zip(D.farm, D.day)]
ed = D.set_index(["farm", "day"]).e_day
D["sib_e"] = [np.mean([ed[x] for x in ss]) if ss else np.nan for ss in D.sibs]

# ---------------- A
cols = ["e_day", "tin", "tin_min", "in_amp", "tout", "rad", "hum", "co2", "heat", "heat_n", "therm", "therm_n", "vent",
        "shade", "circ", "co2a", "fog", "sub_in", "sub_in_n", "sub_amp", "sib_e"]
top_pos = D.nlargest(10, "e_day"); top_neg = D.nsmallest(10, "e_day")
small = D.loc[D.e_day.abs().nsmallest(20).index]
p("A. 하루 프로필 중앙값 (과대예측 상위10 / 과소예측 상위10 / 오차 최소20 / 전체400)")
T = pd.DataFrame({"과대10": top_pos[cols].median(), "과소10": top_neg[cols].median(),
                  "최소20": small[cols].median(), "전체": D[cols].median()})
p(T.round(2).to_string())
p("\n과대예측 상위10:", [(f, d, round(e, 2), pas) for f, d, e, pas in zip(top_pos.farm, top_pos.day, top_pos.e_day, top_pos["pass"])])
p("과소예측 상위10:", [(f, d, round(e, 2), pas) for f, d, e, pas in zip(top_neg.farm, top_neg.day, top_neg.e_day, top_neg["pass"])])
p("형제일 있는 비율: 과대10", top_pos.sib_e.notna().mean(), "과소10", top_neg.sib_e.notna().mean(), "최소20", small.sib_e.notna().mean())
p("큰 오차일의 형제일 오프셋:", [(f, d, round(e, 2), [(x, y, round(ed[(x, y)], 2)) for x, y in ss])
                        for f, d, e, ss in zip(pd.concat([top_pos, top_neg]).farm, pd.concat([top_pos, top_neg]).day,
                                               pd.concat([top_pos, top_neg]).e_day, pd.concat([top_pos, top_neg]).sibs) if ss])

# 오프셋과 '라벨 기반' sub_in 관계 (정의상 강함 — 참고), 입력만의 순위상관 (온실 내)
p("\n전체 400일, 하루 오프셋과 하루 변수 스피어만 (온실별)")
for c in cols[1:]:
    if c in ("sib_e",):
        continue
    r = [spearmanr(D.loc[D.farm == fm, c], D.loc[D.farm == fm, "e_day"], nan_policy="omit")[0] for fm in ("F13", "F47")]
    p(f"  {c:9s} F13 {r[0]:+.3f}  F47 {r[1]:+.3f}")

# ---------------- B 같은 온실 형제 짝: 차이 대 차이
pairs = []
for (f, d), ss in zip(zip(D.farm, D.day), D.sibs):
    for (f2, d2) in ss:
        if f2 == f and d < d2:
            pairs.append((f, d, d2))
P = pd.DataFrame(pairs, columns=["farm", "d1", "d2"])
Di = D.set_index(["farm", "day"])
dcols = ["e_day", "tin", "tin_min", "in_amp", "hum", "co2", "heat", "heat_n", "therm", "therm_n", "vent", "shade", "circ",
         "co2a", "fog", "sub_in", "sub_in_n"]
for c in dcols:
    P["d_" + c] = Di.loc[list(zip(P.farm, P.d1)), c].values - Di.loc[list(zip(P.farm, P.d2)), c].values
p(f"\nB. 같은 온실 형제 짝 {len(P)}개 (F13 {int((P.farm=='F13').sum())}, F47 {int((P.farm=='F47').sum())}), 간격 중앙 {int((P.d2-P.d1).median())}일")
p(f"  짝 안 오프셋 차 |Δe| 중앙 {P.d_e_day.abs().median():.3f} (전체 오프셋 |e| 중앙 {D.e_day.abs().median():.3f})")
p("  Δ오프셋 vs Δ입력 스피어만 (검정 %d개, 서술용):" % (len(dcols) - 1))
for c in dcols[1:]:
    r, pv = spearmanr(P["d_" + c], P.d_e_day, nan_policy="omit")
    p(f"    Δ{c:9s} ρ {r:+.3f} (p {pv:.3f})")
p("  짝 순서(앞 일차 − 뒤 일차) Δ오프셋 부호: 양 %d / 음 %d" % ((P.d_e_day > 0).sum(), (P.d_e_day < 0).sum()))

# ---------------- C 시간별 나란히
p("\nC. 오차 최대 3일과 형제일 시간별 (in_temp / sub_temp / pred / heat / therm / vent / circ)")
worst = D.reindex(D.e_day.abs().sort_values(ascending=False).index).head(6)
shown = 0
for f, d, ss in zip(worst.farm, worst.day, worst.sibs):
    if shown >= 3:
        break
    keys = [(f, d)] + [x for x in ss if x[0] == f][:1]
    if len(keys) < 2:
        keys += [x for x in ss][:1]
    shown += 1
    blocks = []
    for (ff, dd) in keys:
        q = lab[(lab.farm == ff) & (lab.day == dd)].set_index("hour")[["in_temp", "sub_temp", "pred", "act_heating", "act_thermal", "act_vent", "act_circfan"]]
        q.columns = [f"{ff}{dd}_{c[:4]}" for c in ["tin", "sub", "pred", "heat", "ther", "vent", "circ"]]
        blocks.append(q)
    p(f"\n--- {keys} 오프셋 {[round(ed[k], 2) for k in keys]}")
    p(pd.concat(blocks, axis=1).round(1).to_string())
