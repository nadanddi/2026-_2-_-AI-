# -*- coding: utf-8 -*-
# hx8/forensics fx4 v1: 실내 센서 고유 흔적(고착·천장·바닥·자릿수)으로 물리적 동 식별이 되는가 — 서술(학습 기록끼리만).
# 실행 전 고정한 가설 표
# | H2 실내 센서는 동마다 고유한 정적 흔적(습도 천장값·천장 고착 시간, CO2 바닥값, 자릿수 해상도)을 가진다 |
#   흔적: 같은 동 다음날 쌍(진단 연결 c1 ≤ 1)의 흔적 차가 '같은 다음날 묶음의 다른 동' 쌍보다 작음, AUC ≥ .80 |
#   보이면 안 되는 것: AUC ≈ .5~.6(6.599 메타 .50~.55 수준) |
#   추가(ID 전환 검사): 다른 ID로 이어지는 진단 연결(fa≠fb)에서 흔적이 연결 쪽(물리 동)을 따르면 ID ≠ 센서 세트가 아님을 확인 |
#   6.599 와 차이: 결측·자릿수·짝홀이 아니라 '값의 경계'(천장·바닥·고착 시간)와 쌍 선택형 AUC(같은 날짜 묶음 안 고르기).
# 규정: 학습 기록끼리만(평가 입력 미사용). 서술.
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import numpy as np, pandas as pd
from feat4 import X, IN_, AC, EC, ST, FARM, NUM, N, TRAIN, S3

rid = np.array([f"{FARM[i]}_{NUM[i]:03d}" for i in range(N)])
tr_idx = np.where(TRAIN)[0]
def q(i):
    h = IN_[i, :, 1]; c = IN_[i, :, 2]; t = IN_[i, :, 0]
    hm = np.nanmax(h)
    return dict(hum_max=hm, hum_top=np.nansum(np.abs(h - hm) < .05), hum_min=np.nanmin(h),
                co2_min=np.nanmin(c), co2_night=np.nanmedian(c[:6]), t_night=np.nanmin(t[:6]),
                hum_int=np.nanmean(np.abs(h - np.round(h)) < 1e-6), co2_5=np.nanmean(np.abs(c / 5 - np.round(c / 5)) < 1e-6))
Q = pd.DataFrame([q(i) for i in range(N)], index=rid)
print("quirk summary (F13/F47 train):"); print(Q.iloc[tr_idx].describe().round(2).to_string())
print("hum_max top values:", Q.iloc[tr_idx].hum_max.round(1).value_counts().head(8).to_dict())
# same-date bundles (outdoor 24h identical) among train
key = {}
for i in tr_idx: key.setdefault(np.round(X[i], 1).tobytes(), []).append(i)
bundle = {i: v for v in key.values() for i in v}
t = pd.read_pickle(S3 + r"\truth.pkl")
pos = t[(t.c1 <= 1.0)]
print("\npositive links (c1<=1):", len(pos), " cross-ID:", int((pos.fa != pos.fb).sum()))
feat = list(Q.columns)
recs = []
for _, r in pos.iterrows():
    a, b = int(r.a1), int(r.b)
    if not (TRAIN[a] and TRAIN[b]): continue
    others = [k for k in bundle.get(b, []) if k != b]
    for o in others:
        d = {f"d_{f}": (abs(Q.iloc[a][f] - Q.iloc[o][f]) - abs(Q.iloc[a][f] - Q.iloc[b][f])) for f in feat}  # >0: true closer
        acd = lambda x, y: np.nanmean(np.abs(AC[x] - AC[y]) / (np.nanstd(AC[tr_idx].reshape(-1, 5), 0) + 1e-6))
        d["d_actuator"] = acd(a, o) - acd(a, b)
        d.update(a=rid[a], b=rid[b], o=rid[o], cross=FARM[a] != FARM[b], o_sameID=FARM[o] == FARM[a])
        recs.append(d)
D = pd.DataFrame(recs)
print("pairs (true next-day vs other dong same next-day bundle):", len(D))
res = []
for f in feat + ["actuator"]:
    v = D[f"d_{f}"].dropna()
    res.append(dict(trace=f, n=len(v), auc=((v > 0).mean() + .5 * (v == 0).mean())))
print(pd.DataFrame(res).round(3).to_string(index=False))
# ID switch: cross-ID true links — do traces follow the link (physical) or the ID (other record of a's ID in b's bundle)?
C = D[D.cross & D.o_sameID]
print(f"\ncross-ID links where the decoy is a's own ID next day: n={len(C)} (true = other ID, same physical dong)")
for f in ["hum_max", "co2_min", "co2_night", "t_night", "actuator"]:
    v = C[f"d_{f}"].dropna()
    if len(v): print(f"  {f}: trace closer to physical link {((v > 0).mean() + .5 * (v == 0).mean()):.2f} (n={len(v)})")
