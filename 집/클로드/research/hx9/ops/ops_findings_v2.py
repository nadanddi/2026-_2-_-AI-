# -*- coding: utf-8 -*-
"""HX9-OPS 결과 묶기 v2 (2026-10-13 집 클로드): findings_v1.json(단변량·견고성·Y3) + v5/v6(다변량 기준선·누수 점검) + P4/P5 →
findings_v2.json (보고용). 판정 변경은 한 건뿐이며 보수 방향(사후): chg_inp_co2h signal → weak (사유를 notes 에 명시).
findings_v1.json 은 수정하지 않음."""
import os, sys, json
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
F1 = json.load(open(os.path.join(HERE, "findings_v1.json"), encoding="utf-8"))
V6 = pd.read_csv(os.path.join(HERE, "test_v6_splits.csv"))
KEEP_NONE = ["co2_hours", "co2_am_hours", "fog_hours", "fog_pulses", "sealed_frac", "vent_first_h", "th_open_h", "th_close_h",
             "heat_dayfrac", "knn3_hi_mag", "chg_clean_co2h"]
out = []
for f in F1:
    if f["verdict"] == "none" and f["variable"] not in KEEP_NONE:
        continue
    g = dict(f)
    if g["variable"] == "chg_inp_co2h":
        g["verdict"] = "weak"
        g["notes"] = ("[판정 조정·사후·보수 방향] 사전 규칙상 signal(|r| .458, 사슬 CI [.16,.72], LOCO 이득 +.025)이나 "
                      "① 45개 검정 중 FWER .374·BH q .262 로 다중검정 미통과 ② clean 링크판 r .12 → '같은 동 전날 대비 변화' 해석 불성립 "
                      "③ 링커가 고른 전날 기록의 CO2 수준과 r −.455 → 변화가 아니라 링커 선택 산물일 수 있음 ⇒ weak. | " + g["notes"])
    if g["variable"] == "chg_inp_heath":
        t = V6[V6.model == "lin_heath+sg"].set_index("split")
        g["notes"] += (" | [v6] 전체 행 선형 보정(heath+sg): 고EC R2 " + ", ".join(f"{s} {t.at[s, 'R2_nocorr_hi']:+.2f}" for s in t.index) +
                       " / 보통 날 " + ", ".join(f"{s} {t.at[s, 'R2_nocorr_normal']:+.2f}" for s in t.index) +
                       " → 고EC 쪽은 모든 묶음에서 양수지만 보통 날을 망쳐 전체 R2 는 DIAG10 +.02 외 모두 음수. 고EC 를 미리 알아야 쓸 수 있는 형태.")
    out.append(g)

# step-time (P5) and entry-day (P4) findings — diagnostic analyses (EC events are labels)
out += [
 dict(variable="step_drop_actuator_trace", definition="고EC 35기록의 한 시간 EC 하락 계단(ΔEC ≤ −.20, 19칸·11기록) 시각의 구동기·실내 변화 14종 vs 같은 시각대 위약(|ΔEC|<.05)",
      usable="diagnostic only", n=19, r_level=0.0, r_residual=0.0, verdict="none", mechanism="flush(세척) 사건의 구동기 흔적",
      notes="14개 모두 Bonferroni p = 1.0 (최소 원 p .097 co2_on). 하락 계단 중 F47_132/134/143/231 의 9~11시 하락은 커튼 열림(+30~+71)·일사 증가와 같은 시각 "
            "— 아침 배지 온도 상승 시각과 겹침(일반 지식 표시: 원시 EC 는 온도에 따라 변함 0.020/℃, 6.27 확인). 관리 사건 흔적은 관찰되지 않음. 상관 아님(차이 검정)."),
 dict(variable="step_rise_actuator_trace", definition="한 시간 EC 상승 계단(ΔEC ≥ +.20, 38칸·17기록) 시각의 구동기·실내 변화 vs 위약; 이어 기록 안(9~17시) 비교",
      usable="diagnostic only", n=38, r_level=0.0, r_residual=0.0, verdict="none", mechanism="농축·급액 감소 사건의 구동기 흔적",
      notes="기록 사이 비교에서 fog_on Bonf .042, d_in_temp .126, d_act_co2 .364 였으나 기록 안 비교에서 fog_on p .25(→ 포그 쓰는 늦은 고EC 사슬의 '날 특성'), "
            "d_act_co2 Bonf .099. d_in_temp 기록 안 −3.1 (p .0005)은 상승 계단이 14~18시(커튼 닫힘·일사 감소)에 몰린 시각 효과로 보임 — 기록 안 시각 층화 안 함, 관리 흔적 근거 아님."),
 dict(variable="entry_day_schedule_change", definition="에피소드 진입일(clean 전날 보통 → 오늘 hi10) 4건의 운영 지문 변화(chg_clean_L1, L1am, chg_inp_L1)",
      usable="diagnostic only", n=4, r_level=0.0, r_residual=0.0, verdict="none", mechanism="고EC 진입 = 운영 급변(P4)",
      notes="진입일 지문 변화는 다른 전이보다 오히려 작음(백분위 5~17, L1am 1~22; 다른 전이 중앙값 .045 vs 진입일 .010~.021). 에피소드 안 최대 하락일(F47_136 −.67, F47_157 −.43, F47_134 −.39)도 "
            "지문 변화 .017~.032 로 중앙값 이하 → '운영이 바뀌는 날 EC 가 뛰거나 꺼진다'는 흔적 없음. n=4 라 검정 안 함(서술)."),
]
json.dump(out, open(os.path.join(HERE, "findings_v2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(out), "findings written; verdicts:", pd.Series([o["verdict"] for o in out]).value_counts().to_dict())
