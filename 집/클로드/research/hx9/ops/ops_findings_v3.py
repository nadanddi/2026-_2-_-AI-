# -*- coding: utf-8 -*-
"""HX9-OPS 결과 묶기 v3: findings_v2.json 의 step_drop_actuator_trace 해설 정정(내용 오류: '온도 상승 시각과 겹침'을 원시 EC 온도효과처럼 쓴 것은
방향이 반대 — 온도가 오르면 원시 EC 는 올라야 함). 판정·숫자 변경 없음. findings_v2.json 은 그대로 두고 findings_v3.json 로 저장."""
import os, sys, json
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
HERE = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(HERE, "findings_v2.json"), encoding="utf-8"))
for f in F:
    if f["variable"] == "step_drop_actuator_trace":
        f["notes"] = ("14개 모두 Bonferroni p = 1.0 (최소 원 p .097 co2_on). [관찰] 하락 계단 19칸 중 9칸이 9~13시(F13_241, F47_132/134/143/231)로 커튼 열림(+30~+71)·일사 증가와 같은 시각, 9칸이 19~23시(배지 온도 하강 중; "
                      "하락 −.20~−.33 은 온도효과 약 −.03 보다 훨씬 큼)이지만, 위약도 같은 시각대에서 뽑았으므로 그 시각의 일반적 운영과 구별되지 않음. 이때 배지 온도는 오르는데 EC 는 내려 온도효과(0.020/℃)와 반대 방향. "
                      "[해석·일반 지식 표시] 해 뜬 뒤 첫 급액에 의한 희석과 맞는 모양이나 급액은 관측되지 않아 확인 불가. 관리 사건이 구동기에 남긴 흔적은 관찰되지 않음. 상관 아님(차이 검정).")
json.dump(F, open(os.path.join(HERE, "findings_v3.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(F), "findings -> findings_v3.json")
