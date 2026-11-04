# -*- coding: utf-8 -*-
"""TM1b (2026-11-04 집 클로드; 사용자 질문: 발표 그림 2의 전날 실측 반영률 beta_h가 왜 9시 이후 다시 오르나?)
서술용 진단(채택·제출과 무관). TM1(tm1_anchor_physics_measure_v1.py)과 같은 자료·같은 정의로 beta_h를 다시 재고
 (a) 기록 단위 부트스트랩 95% 구간으로 9시 이후 '상승'이 흔들림 범위 안인지,
 (b) 표본을 넓혀(확신 p >= .5) 모양이 유지되는지,
 (c) 앵커 대신 '오늘 0시의 실제 모델 오차' e0 = y0 - q0 로 잰 기울기(e_h on e0)도 같은 모양인지
     → 같다면 beta_h의 모양은 '전날 23시 정보'가 아니라 '모델 오차가 하루 안에서 얼마나 이어지나'의 성질,
 (d) 9시 이후 오차와 0시 오차의 관계가 밤 상태(보온 커튼 닫힘)와 낮 상태에서 다른지 를 본다.
판정 규칙(실행 전 고정): 9~23시 beta_h의 '상승'은 (9~11시 평균)과 (21~23시 평균)의 차이의 부트스트랩 95% 구간이
 0을 포함하면 '흔들림 범위 안(확인 안 됨)', 포함하지 않으면 '상승 확인'.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u tm1b_beta_hour_shape_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta2", os.path.join(HERE, "ta2_anchor_debiased_v1.py")); A2 = importlib.util.module_from_spec(s); s.loader.exec_module(A2)
LK = os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")
RNG = np.random.default_rng(20261104)
NB = 2000


def slopes(g, E):
    """g: (n,), E: (n,24) 잔차 → 시각별 원점 통과 기울기"""
    out = np.full(24, np.nan)
    for h in range(24):
        m = np.isfinite(g) & np.isfinite(E[:, h])
        if m.sum() >= 3:
            out[h] = (g[m] * E[m, h]).sum() / (g[m] ** 2).sum()
    return out


def boot(g, E):
    n = len(g); B = np.empty((NB, 24))
    for b in range(NB):
        i = RNG.integers(0, n, n); B[b] = slopes(g[i], E[i])
    return B


def show(name, g, E):
    b = slopes(g, E); B = boot(g, E)
    lo, hi = np.nanpercentile(B, 2.5, 0), np.nanpercentile(B, 97.5, 0)
    rise = np.nanmean(B[:, 21:24], 1) - np.nanmean(B[:, 9:12], 1)
    r_lo, r_hi = np.nanpercentile(rise, [2.5, 97.5])
    print(f"\n[{name}] n={len(g)}")
    print("  beta_h:  " + " ".join(f"{h}:{b[h]:.2f}" for h in range(24)))
    print("  95%lo:   " + " ".join(f"{h}:{lo[h]:.2f}" for h in range(24)))
    print("  95%hi:   " + " ".join(f"{h}:{hi[h]:.2f}" for h in range(24)))
    point = np.nanmean(b[21:24]) - np.nanmean(b[9:12])
    verdict = "흔들림 범위 안(확인 안 됨)" if r_lo <= 0 <= r_hi else "상승 확인"
    print(f"  (21~23시 평균) - (9~11시 평균) = {point:+.3f}, 95% 구간 [{r_lo:+.3f}, {r_hi:+.3f}] → {verdict}")
    return b


def main():
    L = pd.read_csv(LK); L = L[L.prev_is_labelled.astype(bool) & L.anchor_st23.notna()].copy()
    L["farm"], L["day"] = L.rec.str[:3], L.rec.str[4:7].astype(int)
    R = A2.A1.frames(); M, cells = R["DIAG10"]
    q = np.mean(cells, 0); M = M.assign(q=q, rec=M.farm + "_" + M.day.map("{:03d}".format))
    P = M.pivot_table(index="rec", columns="hour", values=["sub_temp", "q"])
    L = L[L.rec.isin(P.index)].copy()
    y = P["sub_temp"].loc[L.rec].values; qq = P["q"].loc[L.rec].values
    E = y - qq                                   # 오늘 시각별 모델 오차(정답 - 예측)
    L["y0"], L["q0"] = y[:, 0], qq[:, 0]
    p1 = (L.day < 179).values
    hi8 = (L.p >= .8).values; hi5 = (L.p >= .5).values
    # TM1과 같은 자정 전달 식(1차 확신 기록에서 잼)
    m = p1 & hi8 & np.isfinite(L.dT_air.values) & np.isfinite(L.y0.values)
    Xd = np.c_[np.ones(m.sum()), L.dT_air.values[m]]; c, k = np.linalg.lstsq(Xd, (L.y0 - L.anchor_st23).values[m], rcond=None)[0]
    g = L.anchor_st23.values + c + k * L.dT_air.values - L.q0.values   # 전날 23시 실측으로 추정한 오늘 0시 오차
    e0 = E[:, 0]

    show("(a) TM1과 같은 표본: 1차·확신 p>=.8, 앵커 기울기", g[p1 & hi8], E[p1 & hi8])
    show("(b) 표본 넓힘: 1차·확신 p>=.5, 앵커 기울기", g[p1 & hi5], E[p1 & hi5])
    show("(c1) 같은 표본(a), 앵커 대신 실제 0시 오차 e0 기울기", e0[p1 & hi8], E[p1 & hi8])
    allrec = M.pivot_table(index="rec", columns="hour", values=["sub_temp", "q"])
    Ya, Qa = allrec["sub_temp"].values, allrec["q"].values
    Ea = Ya - Qa; ok = np.isfinite(Ea[:, 0])
    days = np.array([int(r[4:7]) for r in allrec.index])
    show("(c2) 전날 연결 없이 모든 학습 기록(1차), 실제 0시 오차 e0 기울기", Ea[ok & (days < 179), 0], Ea[ok & (days < 179)])
    # (d) 하루 수준 몫: 각 기록의 하루 평균 오차를 빼면 모양이 어떻게 되나
    Ec = Ea - np.nanmean(Ea, 1, keepdims=True)
    show("(d) 모든 1차 기록, 하루 평균 오차를 뺀 뒤의 e0 기울기", Ec[ok & (days < 179), 0], Ec[ok & (days < 179)])
    # 시각별 오차 크기(표준편차)
    sd = np.nanstd(Ea[ok & (days < 179)], 0)
    print("\n[참고] 모든 1차 기록 시각별 모델 오차 표준편차: " + " ".join(f"{h}:{sd[h]:.2f}" for h in range(24)))
    cor = [np.corrcoef(Ea[ok & (days < 179) & np.isfinite(Ea[:, h]), 0], Ea[ok & (days < 179) & np.isfinite(Ea[:, h]), h])[0, 1] for h in range(24)]
    print("[참고] 모든 1차 기록 0시 오차와 h시 오차의 상관: " + " ".join(f"{h}:{c_:.2f}" for h, c_ in enumerate(cor)))


if __name__ == "__main__":
    main()
