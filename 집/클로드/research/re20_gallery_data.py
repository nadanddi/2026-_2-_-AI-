# -*- coding: utf-8 -*-
"""재분석 20 (2026-10-01, 집 클로드): 분리 날 갤러리용 데이터 (과대 22·과소 18·정상 표본 20일 + 형제 기록).
출력 local/re20_gallery_data.json (로컬 전용 — 대회 데이터 공개 금지)."""
import env  # noqa
import json, numpy as np, pandas as pd
from harness import load
_, lab, _ = load()
z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
t = lab.in_temp.values; g = np.where(np.isnan(t), 1, np.clip((t - 8) / 2, 0, 1)); s = "DIAG10"
b = np.nanmean([z[f"{s}__MASK__7"], z[f"{s}__MASK__101"]], 0); c = np.nanmean([z[f"{s}__CODEX__726"], z[f"{s}__CODEX__727"]], 0)
pf = np.load(env.LOCAL + f"/web_tabpfn_v2_temp_{s}.npy").mean(0)
lab = lab.assign(pred=(0.6 - 0.2 * (1 - g)) * b + (0.2 + 0.4 * (1 - g)) * c + 0.2 * g * pf)
lab["e"] = lab.pred - lab.sub_temp
lab = lab.sort_values(["farm", "day", "hour"])
E = lab.groupby(["farm", "day"]).e.mean()
W = lab.groupby(["farm", "day"]).apply(lambda q: tuple(np.round(q[["out_temp", "out_hum", "out_rad", "out_wspd"]].values.ravel(), 1)) if len(q) == 24 else None).dropna()
by = {}
for k, sg in W.items(): by.setdefault(sg, []).append(k)
def sibling(k):
    cand = [x for x in by.get(W.get(k), []) if x != k] if k in W.index else []
    same = [x for x in cand if x[0] == k[0]]
    return (same or cand or [None])[0]
grp = {}
for k, e in E.items():
    grp[k] = "과대" if e > 0.7 else "과소" if e < -0.7 else ("정상" if abs(e) < 0.2 else None)
normal = [k for k, v in grp.items() if v == "정상"]
rng = np.random.default_rng(0)
pick = set(k for k, v in grp.items() if v in ("과대", "과소")) | set(map(tuple, rng.permutation(np.array(normal, dtype=object))[:20]))
cols = {"in_temp": "air", "sub_temp": "sub", "pred": "pred", "act_heating": "heat", "act_thermal": "therm", "act_vent": "vent", "act_circfan": "circ", "out_temp": "out"}
def series(k):
    q = lab[(lab.farm == k[0]) & (lab.day == k[1])].set_index("hour").reindex(range(24))
    return {v: [None if pd.isna(x) else round(float(x), 2) for x in q[c].values] for c, v in cols.items()}
days = []
for k in sorted(pick, key=lambda k: -abs(E[k])):
    sk = sibling(k)
    d = {"farm": k[0], "day": int(k[1]), "group": grp[k], "offset": round(float(E[k]), 2), "pass": 2 if k[1] >= 179 else 1,
         "s": series(k), "sib": None}
    if sk is not None:
        d["sib"] = {"farm": sk[0], "day": int(sk[1]), "offset": round(float(E[sk]), 2), "s": series(sk)}
    days.append(d)
json.dump(days, open(env.LOCAL + "/re20_gallery_data.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(days), pd.Series([d["group"] for d in days]).value_counts().to_dict(), sum(d["sib"] is not None for d in days))
