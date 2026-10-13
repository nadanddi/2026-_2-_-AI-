# -*- coding: utf-8 -*-
"""HX9-OPS 결과 묶기 v1: test_v1(단변량) + v2(p 결함 수정·견고성) + v4(Y3 사슬 CI) → findings_v1.json (보고용, 판정 변경 없음)."""
import os, sys, json
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
T = pd.read_csv(os.path.join(HERE, "test_v2_univariate_pfix.csv")).set_index("variable")
RB = pd.read_csv(os.path.join(HERE, "test_v2_robust.csv")).set_index("variable")
Y3 = pd.read_csv(os.path.join(HERE, "test_v4_Y3.csv")).set_index("variable")
GROUP = {"co2": "CO2 공급 체제", "th_": "보온커튼 일정", "sh_": "차광 일정", "curtain": "두 커튼 분리 운전", "fog": "포그 펄스",
         "heat": "난방 일정", "vent": "환기 정책", "sealed": "환기 정책(밀폐)", "fan": "유동팬 운전", "op_n": "운영 활동 수",
         "te_": "운영 지문 군집 체제(교차적합 목표 부호화)", "proto": "고EC 운영 지문 원형 유사도", "knn": "운영 지문 최근접 고EC 크기",
         "chg_clean": "전날(clean 링크) 대비 운영 변화", "chg_inp": "전날(입력 링커) 대비 운영 변화"}
def grp(v):
    for k, g in GROUP.items():
        if v.startswith(k): return g
    return "기타"
f = lambda x, d=3: None if (x is None or (isinstance(x, float) and not np.isfinite(x))) else round(float(x), d)
out = []
for v, d in T.iterrows():
    n = int(d.n_Y2h) if np.isfinite(d.n_Y2h) else 0
    note = []
    if not np.isfinite(d.r_Y2h):
        note.append("고EC 31기록 안에서 상수(분산 0) → 상관 정의 안 됨")
    else:
        note.append(f"rho_resid {f(d.rho_Y2h,2)}; 계절+sg 통제 편상관 {f(d.pr_Y2h_season_sg,2)}; 사슬 하나 빼기 r {f(d.loco_r_min,2)}~{f(d.loco_r_max,2)}")
        note.append(f"F13 {f(d.r_Y2h_F13,2)} / F47 {f(d.r_Y2h_F47,2)}; 1차 고EC {f(d.r_Y2h_pass1,2)} / 2차 고EC(n5) rho {f(d.rho_Y2h_pass2_n5,2)}")
        note.append(f"전체 360 잔차 r {f(d.r_Y2a,2)} (1차 {f(d.r_Y2a_pass1,2)}, 2차 {f(d.r_Y2a_pass2,2)}), 전체 CV 이득 {f(d.cv_gain_Y2a,3)}")
        note.append(f"순열 p {f(d.p_perm_Y2h,3)}, FWER(max-T) {f(d.p_fwer_maxT_Y2h,3)}, BH q {f(d.q_bh_Y2h_fixed,3)}; 개선 기록 {int(d.n_improved_Y2h)}/31")
    if v in Y3.index:
        y = Y3.loc[v]; note.append(f"Y3 사슬 CI [{f(y.ci_lo,2)},{f(y.ci_hi,2)}], 날씨변화 통제 {f(y.pr_weather,2)}, Y3 FWER {f(y.p_fwer,2)}")
    note.append(f"하루 안 변화(23−0시) r {f(d.r_Y3i,2)}")
    if v in RB.index and v != "proto_sim_hi":
        b = RB.loc[v]
        note.append(f"[v2 사후 견고성] F47_132 제외 r {f(b.r_drop_F47_132,2)}, clean 링크판 r {f(b.r_clean,2)}, 0~11시판 r {f(b.r_am_0to11,2)}, "
                    f"무작위 전날 위약 p {f(b.placebo_p,3)}, 자기 수준 r {f(b.r_own_level,2)}, 전날 수준 r {f(b.r_pred_level,2)} → 생존 {bool(b.survive)}")
    if v == "proto_sim_hi":
        note.append(f"[v2] F47_132 제외 r {f(RB.loc[v].r_drop_F47_132,2)}; 음의 부호 = 고EC 원형에 가까울수록 모델이 덜 과소 → '전형적 고EC 운영일은 이미 모델이 잡음'")
    if v == "chg_inp_heath":
        note.append("[v3] 전날 대비 외기온 변화와 r −.68~−.73(추워진 날 난방 증가). 외기·일사·실내 밤온도 변화 통제 편상관 .315, +계절+sg .408 → 일부는 날씨 변화 대리")
    if v == "te_clu_mag":
        note.append("자기 사슬 제외 평균이라 Y1 과 음의 상관(−.75)은 부호화 인공물")
    out.append(dict(variable=v, definition=f"{d.definition} (평가 행 확정 시각 avail_h={d.avail_h})", usable=d.usable, n=n,
                    r_level=f(d.r_Y1) if np.isfinite(d.r_Y1) else 0.0, r_residual=f(d.r_Y2h) if np.isfinite(d.r_Y2h) else 0.0,
                    r_increment=f(d.r_Y3) if np.isfinite(d.r_Y3) else 0.0,
                    ci_residual=(f"[{f(d.ci_chain_lo,2)}, {f(d.ci_chain_hi,2)}] 사슬 부트스트랩(8사슬); 기록 부트스트랩 [{f(d.ci_rec_lo,2)}, {f(d.ci_rec_hi,2)}]"
                                 if np.isfinite(d.r_Y2h) else "정의 안 됨"),
                    cv_r2_gain_residual=f(d.cv_gain_Y2h) if np.isfinite(d.cv_gain_Y2h) else 0.0,
                    mechanism=grp(v), verdict=d.verdict, notes=" | ".join(note)))
json.dump(out, open(os.path.join(HERE, "findings_v1.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(out), "findings written")
