# 집 최종 v4 생성기 → v5 (2026-11-04, 집 클로드). 사용자 수정 목록 1~4:
#  1. 1.2를 세 장으로((1/3) 찾은 점 ① 표 + 실제 예시 하루 그래프, (2/3) ② 그림 크게, (3/3) ③④ 막대그래프 둘)
#  2. 1.4 글머리표 3줄, 그림 반영률 캡션 정정, 대사 한 문장, 노트, 4 정리 향후 계획 한 줄
#  3. 1.5 EC를 온도와 같은 틀로(세 단계 표 + 반영률 비교 그림), 꼬리표 '근거: 찾은 점 ②', 대사 한 문장
#  4. 1.3·1.5 표에 '모델 종류' 열(계열 이름), '물리 기준 예측' 캡션·대사 삭제
# 바꿀 원문이 정확히 한 번 있는지 확인하고, 하나라도 어긋나면 파일을 쓰지 않는다.
import json
import os
import re
import sys

PATH = sys.argv[1]
s = open(PATH, encoding="utf-8").read()
EX = json.load(open(os.path.join(os.path.dirname(os.path.abspath(PATH)), "data", "example_day_v1.json"), encoding="utf-8"))


def need(c, m):
    if not c:
        sys.exit("실패: " + m)


def rep1(t, a, b):
    need(t.count(a) == 1, f"원문 {t.count(a)}곳: {a[:70]!r}")
    return t.replace(a, b)


def block(t, key):
    i = t.index("// ================= " + key)
    j = t.index("\n// =================", i + 5)
    return i, j


def notes_expr(b):
    m = re.search(r"\n  s\.addNotes\((.*)\);\n", b)
    need(m, "노트 줄")
    return m.group(1)


def get_said(e):
    i = e.index("[말할 것] ") + len("[말할 것] ")
    return e[i: e.index("\\n\\n", i)]


def set_said(e, said):
    i = e.index("[말할 것] ")
    j = e.index("\\n\\n", i)
    return e[:i] + "[말할 것] " + said + e[j:]


def js(t):
    return t.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# ======== 1. 1.2 → 세 장 ========
i, j = block(s, "1.2 데이터에서 찾은 점")
old12 = s[i:j]
n12 = notes_expr(old12)
said = get_said(n12)
need(said.startswith("데이터에서 네 가지를 찾았습니다. 첫째, "), "1.2 대사 시작")
s1 = ("데이터에서 네 가지를 찾았습니다. 첫째, 하루 전체가 통째로 밀리는 오차입니다. 오른쪽 실제 하루처럼, 예측이 하루 내내 거의 같은 간격으로 낮습니다. "
      "보정 전 기준 모델로 재 보니, F13과 F47 모두 전체 오차 중 절반가량이 이런 밀림이었고, 그날 센서 요약으로는 이 밀림을 맞히지 못했습니다.")
s2 = "둘째, 어떤 기록의 23시 배지 값이 다음 기록의 0시 값으로 끊김 없이 이어졌고, 확실한 짝 190쌍 중 40쌍은 다른 ID에 실려 있었습니다."
s3 = "셋째, 사전학습 모델 C는 실내 8도 이하에서 오차가 B의 두 배를 넘었습니다. 넷째, EC가 높은 동은 추운 날 배지 온도를 낮게 예측했습니다."
need("둘째, 어떤 기록의 23시 배지 값이 다음 기록의 0시 값으로 끊김 없이 이어졌고, 확실한 짝 190쌍 중 40쌍은 다른 ID에 실려 있었습니다." in said, "1.2 둘째 문장")

ex_note = (f"[예시 하루(집 v5)] {EX['record']}: DLV1과 같은 기준 모델(선형 회귀 + LightGBM 잔차, 날짜 연속 5덩어리 OOF)의 예측. "
           f"고른 규칙(실행 전 고정): 하루 제곱 오차 합이 그 온실 상위 10%인 날 중 그날 안 하루 밀림 몫(24·m²/Σe²)이 가장 큰 날, F13·F47 중 더 큰 쪽. "
           f"이 날의 하루 밀림 몫 {EX['within_day_level_share']}, 평균 오차(예측−정답) {EX['mean_error']}℃. "
           "표의 F13·F47 수치(.486/.588, .529/.487)는 같은 계산으로 재현됨. 스크립트 집/클로드/research/dlv1d_example_day_v1.py, 로그 logs/dlv1d_v1.log, 자료 집/클로드/발표/최종/data/example_day_v1.json. "
           "밀림이 가장 뚜렷한 날을 고른 예시이며 모든 날이 이렇지는 않음.")
H = [str(h) if h % 3 == 0 else "" for h in EX["hour"]]
new1 = f'''// ================= 1.2 데이터에서 찾은 점 (1/3) ① 하루 전체 밀림 (집 v5) =================
{{
  const s = slide("1.2", "데이터에서 찾은 점 (1/3)", "① 기준 모델 오차의 절반은 '하루 전체 밀림'", "출처: DLV1(카탈로그 6.997·7.078), 예시 하루는 같은 기준 모델로 다시 계산(dlv1d_example_day_v1.py). 학습 자료에서 잰 값.");
  table(s, [
    ["무엇을 확인했나", "F13", "F47", "49곳 중앙값"],
    ["전체 오차 중 하루 밀림의 몫", "49%", "53%", "49%"],
    ["가장 나쁜 10% 날의 몫", "59%", "49%", "54%"],
    ["그날 센서로 밀림을 맞힌 정도(설명력)", "−0.18", "−0.12", "−0.64"],
  ], {{ x: 0.9, y: 1.95, w: 5.7, colW: [2.9, 0.8, 0.8, 1.2], fontSize: 13, rowH: 0.62 }});
  cap(s, "표 2. 보정 전 기준 모델의 교차검증 오차(제곱 오차 기준). 설명력 음수 = 평균값보다 못 맞힘", {{ x: 0.9, y: 4.5, w: 5.7, h: 0.55 }});
  const HH = {json.dumps(H, ensure_ascii=False)};
  s.addChart(pres.charts.LINE, [
    {{ name: "정답", labels: HH, values: {json.dumps(EX["truth"])} }},
    {{ name: "기준 모델 예측", labels: HH, values: {json.dumps(EX["pred"])} }},
  ], Object.assign({{ x: 6.85, y: 1.7, w: 5.8, h: 2.75, chartColors: [K.ink, K.acc], lineSize: 2.5, lineDataSymbol: "none", showLegend: true, legendPos: "t", legendFontFace: SANS, legendFontSize: 11, legendColor: K.sub,
    valAxisMinVal: 12, valAxisMaxVal: 26, valAxisMajorUnit: 2, valAxisLabelFormatCode: "0", objectName: nm("chart") }}, AX,
    {{ showTitle: true, title: "그림 1. 실제 하루({EX['record']}) 배지 온도(℃)", titleFontSize: 12, titleColor: K.ink, titleFontFace: SANS }}));
  s.addChart(pres.charts.LINE, [
    {{ name: "실내 온도", labels: HH, values: {json.dumps(EX["in_temp"])} }},
    {{ name: "외기 온도", labels: HH, values: {json.dumps(EX["out_temp"])} }},
  ], Object.assign({{ x: 6.85, y: 4.5, w: 5.8, h: 1.9, chartColors: [K.acc2, K.mute], lineSize: 1.75, lineDataSymbol: "none", showLegend: true, legendPos: "r", legendFontFace: SANS, legendFontSize: 10, legendColor: K.sub,
    valAxisMinVal: 0, valAxisMaxVal: 30, valAxisMajorUnit: 10, valAxisLabelFormatCode: "0", objectName: nm("chart") }}, AX,
    {{ showTitle: true, title: "같은 날 실내·외기 온도(℃), 가로축 = 시각", titleFontSize: 11, titleColor: K.sub, titleFontFace: SANS }}));
  cap(s, "예시: 밀림이 가장 뚜렷한 날을 미리 정한 규칙으로 고름. 예측이 하루 평균 {abs(EX['mean_error']):.1f}℃ 낮음", {{ x: 6.95, y: 6.42, w: 5.7, h: 0.4 }});
  s.addNotes({set_said(n12, s1)} + "\\n\\n{js(ex_note)}");
}}
'''
fig_old = re.search(r"  // 그림 1\(옛 1\.1에서 옮김\).*?\n  cap\(s, \"그림 1\. 오늘\(3일, F47\)의 전날\(2일\)이 다른 ID\(F13\)에 실린 예\(개념도, 날짜는 예시\)\", \{ x: 7\.5, y: 4\.2, w: 5\.1, h: 0\.55 \}\);\n", old12, re.S)
need(fig_old, "그림 1 코드")
fig2 = '''  // 그림(옛 1.1 그림 1): 같은 동의 하루하루가 두 ID를 오감 — 집 v5에서 크게
  const lane = (y, id) => { T(s, id, { x: 1.4, y: y + 0.2, w: 0.9, h: 0.5, fontSize: 18, bold: true, color: K.ink }); hline(s, 2.3, y + 0.45, 9.6, K.hair, 0.75); };
  lane(2.3, "F13"); lane(3.9, "F47");
  const days = [[0, "1일"], [0, "2일"], [1, "3일"], [1, "4일"], [0, "5일"], [1, "6일"]];
  days.forEach((d, i) => {
    const x = 2.6 + i * 1.55, y = d[0] ? 3.9 : 2.3;
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 1.0, h: 0.9, fill: { color: i === 2 ? K.acc : K.sand }, line: { color: i <= 2 && i >= 1 ? K.acc : K.sand, width: i === 1 ? 2.5 : 0.5 }, objectName: nm("day") });
    T(s, d[1], { x, y, w: 1.0, h: 0.9, align: "center", valign: "middle", fontSize: 16, bold: true, color: i === 2 ? "FFFFFF" : K.ink });
    if (i < days.length - 1) {
      const ny = days[i + 1][0] ? 3.9 : 2.3;
      const y1 = y + 0.45, y2 = ny + 0.45;
      s.addShape(pres.shapes.LINE, { x: x + 1.0, y: Math.min(y1, y2), w: 0.55, h: Math.abs(y2 - y1), flipV: y2 < y1, line: { color: i === 1 ? K.acc : K.sub, width: i === 1 ? 2.5 : 1.25, endArrowType: "triangle" }, objectName: nm("link") });
    }
  });
'''
new2 = f'''// ================= 1.2 데이터에서 찾은 점 (2/3) ② 이어지는 전날 기록 (집 v5) =================
{{
  const s = slide("1.2", "데이터에서 찾은 점 (2/3)", "② 전날 기록은 이어지고, 일부는 다른 ID에 실림", "출처: XID1 재계산(2026-11-03, 카탈로그 7.072). 짝 판정 근거는 1.1 노트.");
{fig2}  cap(s, "그림 2. 오늘(3일, F47)의 전날(2일)이 다른 ID(F13)에 실린 예(개념도, 날짜는 예시)", {{ x: 1.4, y: 5.05, w: 10.5, h: 0.35 }});
  T(s, "23시 → 0시 배지 값이 끊김 없이 이어지는 확실한 짝 190쌍 중 40쌍(21%)은 다른 ID", {{ x: 1.4, y: 5.55, w: 10.5, h: 0.45, fontSize: 17, bold: true, color: K.ink }});
  s.addNotes("[말할 것] {js(s2)}\\n\\n[노트] (집 v5: 옛 1.2 그림 1을 따로 크게) 확실한 짝 = 앞 기록 23시와 뒤 기록 0시의 배지 EC·온도 정답 차이가 작고 두 번째 후보보다 확실히 가까우며 외기도 이어지는 1:1 짝(linker_v3/t3.py). 190쌍 중 40쌍이 다른 ID(F13→F47 19, F47→F13 21; 집/클로드/research/xid1_v1.log). 같은 동의 기록이 두 ID에 나뉘어 실린 것으로 해석하지만, 원천 동 ID·수집 이력이 없어 원인은 확정 못 함. 상세는 1.1 노트.");
}}
'''
c3 = ("[노트] (집 v5 새 장)\\n"
      "- ③ 카탈로그 6.67: 실내 8℃ 이하 164행(추운 날 검증 EXT12)의 RMSE. TabPFN 단독 2.35~2.45℃(편향 +1.8~+1.9), Codex 단독 0.942℃. 막대는 범위의 가운데 값 2.40. 따뜻할 때 TabPFN을 빼면 일반 날 검증(DIAG10) 오차가 약 3.0~3.3% 늘어남(같은 항목, 'TabPFN 없는 모델 대비'). 개발 당시 구성(10월 초)의 값이며 최종 모델 성능이 아님.\\n"
      "- ④ 카탈로그 6.799(TX1): 13회차 구성의 온도 잔차(정답−예측) 평균. 최저 10℃ 미만인 날 검증 EXT10에서 고EC 동 +0.64℃, 보통 동 약 0(카탈로그 기록 '≈0'), 최저 12℃ 기준 EXT12 고EC 동 +0.44℃. 이것이 1.4의 추운 조건 보정(TX3, 최대 +0.25℃)의 근거.\\n"
      "- 두 값 모두 학습 자료 검증에서 잰 값.")
new3 = f'''// ================= 1.2 데이터에서 찾은 점 (3/3) ③④ 추위 (집 v5) =================
{{
  const s = slide("1.2", "데이터에서 찾은 점 (3/3)", "③ 모델 C는 추위에 약하고, ④ 고EC 동은 추운 날 낮게 예측", "출처: ③ 카탈로그 6.67, ④ 카탈로그 6.799. 둘 다 개발 당시 구성에서 학습 자료로 잰 값(최종 모델 성능 아님).");
  T(s, "③ 사전학습 모델 C는 학습 범위 밖 추위에서 크게 틀림", {{ x: 0.9, y: 1.75, w: 5.6, h: 0.45, fontSize: 15, bold: true, color: K.ink }});
  bar(s, ["C만 사용", "B만 사용"], [["오차", [2.40, 0.94]]], {{ x: 0.8, y: 2.25, w: 5.6, h: 3.0, chartColors: [K.acc], valAxisMinVal: 0, valAxisMaxVal: 3, valAxisMajorUnit: 1, dataLabelFormatCode: "0.00",
    showValAxisTitle: true, valAxisTitle: "오차 RMSE(℃)", valAxisTitleFontSize: 11, valAxisTitleColor: K.sub, valAxisTitleFontFace: SANS }});
  cap(s, "그림 3. 실내 8℃ 이하 164시간의 오차. C는 2.35~2.45(가운데 값 표시). 따뜻할 때는 C가 도움: 빼면 일반 날 오차 약 3% 증가", {{ x: 0.9, y: 5.3, w: 5.6, h: 0.75 }});
  T(s, "④ EC가 높은 동은 추운 날 배지 온도를 낮게 예측", {{ x: 7.0, y: 1.75, w: 5.6, h: 0.45, fontSize: 15, bold: true, color: K.ink }});
  bar(s, ["고EC 동", "보통 동"], [["정답−예측", [0.64, 0.0]]], {{ x: 6.9, y: 2.25, w: 5.6, h: 3.0, chartColors: [K.acc2], valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.2, dataLabelFormatCode: '+0.00;-0.00;"≈0"',
    showValAxisTitle: true, valAxisTitle: "정답 − 예측 평균(℃)", valAxisTitleFontSize: 11, valAxisTitleColor: K.sub, valAxisTitleFontFace: SANS }});
  cap(s, "그림 4. 최저 10℃ 미만인 날 검증. 보통 동은 약 0. 최저 12℃ 기준에서도 고EC 동 +0.44", {{ x: 7.0, y: 5.3, w: 5.6, h: 0.75 }});
  s.addNotes("[말할 것] {js(s3)}\\n\\n{c3}");
}}
'''
s = s[:i] + new1 + "\n" + new2 + "\n" + new3 + s[j:]

# ======== 4. 1.3 모델 종류 열, '물리' 문장 삭제 ========
i, j = block(s, "1.3 온도 기본 예측")
b = s[i:j]
b = rep1(b, '''    ["모델", "무엇을 보나"],
    ["A", "실내외 센서와 난방·환기 기록, 앞선 며칠의 변화까지 함께 봄"],
    ["B", "오늘 0시부터 지금까지 센서와 구동기가\\n어떻게 변했는지 봄"],
    ["C", "학습 기록들과 그 배지 온도를 참고해 예측\\n(공개 사전학습 모델, 재학습 없음)"],
  ], { x: 0.9, y: 1.95, w: 6.9, colW: [0.9, 6.0], fontSize: 15, rowH: [0.5, 0.8, 0.8, 0.8] });''',
'''    ["모델", "무엇을 보나", "모델 종류"],
    ["A", "실내외 센서와 난방·환기 기록,\\n앞선 며칠의 변화까지 함께 봄", "부스팅 트리\\n+ 선형 회귀"],
    ["B", "오늘 0시부터 지금까지 센서와\\n구동기가 어떻게 변했는지 봄", "부스팅 트리\\n+ 선형 회귀"],
    ["C", "학습 기록들과 그 배지 온도를 참고해\\n예측(재학습 없음)", "사전학습\\n트랜스포머"],
  ], { x: 0.9, y: 1.95, w: 6.9, colW: [0.8, 4.3, 1.8], fontSize: 14, rowH: [0.5, 0.85, 0.85, 0.85] });''')
ne = notes_expr(b)
said = get_said(ne)
need(said.count(" 둘 다 물리 기준 예측에 학습한 보정을 더합니다.") == 1, "1.3 대사 물리 문장")
said = said.replace(" 둘 다 물리 기준 예측에 학습한 보정을 더합니다.", "")
b = b.replace(ne, set_said(ne, said) + ' + "\\n\\n[모델 종류(집 v5)] A(BASE) = 물리 선형식(실내 온도·일사·난방·외기의 지수 가중 평균, 반감기 1~72시간, 14열)을 Ridge로 맞추고 남은 오차를 LightGBM(huber) 0.65·Ridge 0.25·Nystroem 커널 0.10으로 보정. B(CODEX) = Ridge 기준선 + LightGBM 잔차. C = TabPFN v2(합성 표 데이터로 사전학습된 트랜스포머, 재학습 없음). 개발 중 GRU(RNN)를 시험했으나 최종 구성에는 없음. 근거: 정형 README ② 온도 1, features_v4.py phys_features."')
s = s[:i] + b + s[j:]

# ======== 2. 1.4 ========
i, j = block(s, "1.4 온도 보정")
b = s[i:j]
b = rep1(b, '''  T(s, bl(st.slice(0, 2).map((v) => v[0] + ": " + v[1].replace(/\\n/g, " ")), 15, K.body, 10), { x: 0.9, y: 2.42, w: 6.0, h: 2.4 });''',
'''  T(s, bl([  // 집 v5: 세 줄로 다시 씀
    "전날 찾기: 23시→0시 외기·실내·구동기 흐름이 이어지는 기록을 전날로 봄. 짝이 확실할 때만 사용",
    "전날 실측이 있으면: 그 값으로 오늘 0시 예측의 빗나감을 추정해, 새벽에 크게·낮에 작게 더함",
    "전날 실측이 없으면: 전날에 대한 우리 예측값을 대신 쓰되, 더 약하게 더함",
  ], 15, K.body, 10), { x: 0.9, y: 2.42, w: 6.0, h: 2.5 });''')
b = rep1(b, '"그림 2. 전날 실측 보정 반영률은 새벽에 크다"', '"그림 5. 전날 실측 보정 반영률(제출 모델 값)"')
b = rep1(b, "    x: 7.15, y: 2.25, w: 5.5, h: 3.6, chartColors: [K.acc],", "    x: 7.15, y: 2.25, w: 5.5, h: 3.3, chartColors: [K.acc],")
b = b.replace("\n}\n", '\n  cap(s, "0시 76% → 9시 13%로 빠르게 줄어듦. 9시 이후 오르내림은 학습 기록 32개로 잰 흔들림이며, 큰 표본(318개)에서는 계속 줄어 23시 약 8%", { x: 7.25, y: 5.6, w: 5.4, h: 0.7 });\n}\n', 1)
ne = notes_expr(b)
said = get_said(ne)
o = "실측이 없으면 쓸 수 있는 전날 예측을 줄여 반영합니다."
need(said.count(o) == 1, "1.4 대사")
said = said.replace(o, "실측이 없으면 전날에 대한 우리 예측값을 대신 쓰되, 더 약하게 반영합니다.")
an = ("[반영률의 뜻과 모양(집 v5)] 그림 5 = 제출 모델의 β_h. 보정 pred_h = q_h + β_h·clip(A − 0.36 − q_0, −2, 2): 전날 23시 실측 A로 추정한 오늘 0시 빗나감에 곱하는 비율. "
      "β_h는 확신 높은 학습 기록 32개로 시각마다 따로 잰 값(카탈로그 6.583·6.744). 2026-11-04 다시 잼(집/클로드/research/tm1b_beta_hour_shape_v1.py, logs/tm1b_v1.log): "
      "32개 기준 '21~23시 평균 − 9~11시 평균' = +0.04, 기록 부트스트랩 95% 구간 [−0.12, +0.22] → 9시 이후 상승은 확인 안 됨(66개로 넓혀도 같음). "
      "전날 연결 없이 학습 기록 318개 전체로 '오늘 0시 실제 모델 오차가 h시 오차에 남는 몫'을 재면 0시 1.00 → 9시 0.30 → 23시 0.08로 하루 내내 줄고, 저녁−아침 = −0.18, 구간 [−0.28, −0.06]. "
      "해석: 9시 이후의 작은 몫(약 1할)은 하루 단위 밀림 때문에 남는 것이 맞지만, 23시 24%는 표본 흔들림이 그대로 들어간 값. 매끄럽게 줄어드는 곡선과의 비교는 하지 않음(향후 계획).")
b = b.replace(ne, set_said(ne, said) + ' + "\\n\\n' + js(an) + '"')
b = b.replace("1.4 그림 2의 세로축", "1.4 그림 5의 세로축").replace("- 그림 2 = β_h", "- 그림 5 = β_h")
s = s[:i] + b + s[j:]

# ======== 3. 1.5 EC ========
i, j = block(s, "1.5 EC 모델")
b = s[i:j]
ne = notes_expr(b)
said = get_said(ne) + " 온도와 달리 EC는 전날 실측을 하루 내내 같은 비율로 반영합니다."
BETA = "[0.7648, 0.6232, 0.5525, 0.4844, 0.4091, 0.3453, 0.3025, 0.2576, 0.1866, 0.1264, 0.157, 0.1661, 0.1655, 0.1566, 0.1999, 0.2032, 0.2233, 0.2263, 0.2145, 0.2008, 0.1897, 0.1721, 0.1991, 0.2355]"
ecn = ("[EC 구성(집 v5, 정형 README ② EC)] ① 기본 예측 pre = clip(일내 수축(0.8·R3 + 0.2·TabPFN)), R3 = ExtraTrees 0.6 + LightGBM 0.3 + MLP 0.1(시드 3개), TabPFN v2 문맥 2,000행×4. "
       "입력: 그 시각 실내·구동기, 0시 값, 0..h시 누적 평균·0 비율, 운영 순서(밀폐 연속·환기 누적·커튼 전환 수), 계절 지표. "
       "② 하루 밀림 조정 HR5: 0·3…21시마다 TabPFN v2가 그 기록의 하루 평균 잔차를 예측, NH = pre + 0.5·보정. "
       "③ 전날 EC 앵커: pred_h = q_h + 0.8·p·(A − mean(q_0..q_h)), A = 연결된 전날 학습 기록의 23시 EC 정답, 대상은 온도 앵커와 같은 목록(p ≥ .8 + TAG2). "
       "그림 6의 EC 선은 p = 1일 때 0.8(80%). 온도 선은 그림 5와 같은 값. EC의 하루 내내 같은 비율은 당시 검증과 제출 비교(13회차 EA2 '하루 종일 수준 유지')로 정함. "
       "꼬리표: 찾은 점 ①(하루 밀림)은 배지 온도로 잰 값이라 EC 근거로는 ②만 적음.")
newEC = f'''// ================= 1.5 EC 모델 (집 v5: 온도와 같은 틀) =================
{{
  const s = slide("1.5", "EC 모델 · 근거: 찾은 점 ②", "기본 예측 → 하루 밀림 조정 → 전날 실측 보정", "식: 정형 README ②(EC, 122~138줄). ② = 하루 밀림 조정(HR5), ③ = EC 실측 보정(앵커).");
  table(s, [
    ["단계", "무엇을 보나", "모델 종류"],
    ["① 기본 예측", "그 시각까지의 실내·구동기 기록과\\n운영 순서(밀폐·환기·커튼)", "트리 앙상블 + 신경망\\n+ 사전학습 트랜스포머"],
    ["② 하루 밀림 조정", "3시간마다 오늘 하루 평균이 얼마나\\n빗나갈지 예측해 절반 반영", "사전학습 트랜스포머"],
    ["③ 전날 실측 보정", "전날 23시 EC 실측과의 차이를 하루 내내\\n같은 비율로 반영(짝이 확실할 때만)", "보정식"],
  ], {{ x: 0.9, y: 1.95, w: 6.9, colW: [1.55, 3.25, 2.1], fontSize: 12, rowH: [0.5, 0.95, 0.95, 0.95] }});
  cap(s, "표 5. EC 예측 세 단계", {{ x: 0.9, y: 5.4, w: 6.9, h: 0.35 }});
  const B = {BETA};
  s.addChart(pres.charts.LINE, [
    {{ name: "온도", labels: B.map((_, h) => (h % 3 === 0 ? String(h) : "")), values: B.map((v) => Math.round(v * 1000) / 10) }},
    {{ name: "EC", labels: B.map((_, h) => (h % 3 === 0 ? String(h) : "")), values: B.map(() => 80) }},
  ], Object.assign({{ x: 8.05, y: 1.85, w: 4.6, h: 3.5, chartColors: [K.acc, K.acc2], lineSize: 2.5, lineDataSymbol: "none", showLegend: true, legendPos: "t", legendFontFace: SANS, legendFontSize: 11, legendColor: K.sub,
    valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 20, valAxisLabelFormatCode: "0", objectName: nm("chart") }}, AX,
    {{ showTitle: true, title: "그림 6. 전날 실측 반영률(%)", titleFontSize: 12, titleColor: K.ink, titleFontFace: SANS,
      showCatAxisTitle: true, catAxisTitle: "오늘 시각(시)", catAxisTitleFontSize: 11, catAxisTitleColor: K.sub, catAxisTitleFontFace: SANS }}));
  cap(s, "온도는 새벽에만 크게, EC는 하루 내내 같은 비율(짝 확신도 1일 때 80%). 둘 다 제출 모델 값", {{ x: 8.1, y: 5.4, w: 4.55, h: 0.6 }});
  s.addNotes({set_said(ne, said)} + "\\n\\n{js(ecn)}");
}}
'''
s = s[:i] + newEC + s[j:]

# ======== 2. 4 정리: 향후 계획 한 줄 ========
s = rep1(s, '"달력으로 계절 지표 교체\\n실제 전날로 재검증"', '"달력으로 계절 지표 교체\\n실제 전날로 재검증\\n전날 반영률을 줄어드는 곡선으로 다시 맞추기"')

# ======== 그림·표 번호 다시 매기기(본문 캡션·제목 순서대로) ========
body_end = s.index("// ================= 부록")
body = s[:body_end]
order = {"그림": 0, "표": 0}


def renum(m):
    order[m.group(1)] += 1
    return f'"{m.group(1)} {order[m.group(1)]}.'


body = re.sub(r'"(그림|표) (\d+)\.', renum, body)
body = re.sub(r'title: "(그림) (\d+)\.', lambda m: (order.__setitem__("그림", order["그림"]) or m.group(0)), body)
s = body + s[body_end:]

# 머리 주석·출력
s = rep1(s, "// 팜모니 발표자료 — 집 최종 v4 (2026-11-04, 집 클로드)",
         "// 팜모니 발표자료 — 집 최종 v5 (2026-11-04, 집 클로드) — v4 + 사용자 수정 목록 1~4: 1.2를 세 장으로(찾은 점 ① 표 + 실제 예시 하루 F47_189 그래프 / ② 그림 크게 / ③④ 막대그래프), "
         "1.4 글머리표·반영률 캡션 정정, 1.5 EC를 온도와 같은 틀(세 단계 표 + 반영률 비교 그림), 1.3·1.5 '모델 종류' 열, 4 정리 향후 계획 한 줄.\n"
         "// 팜모니 발표자료 — 집 최종 v4 (2026-11-04, 집 클로드)")
s = rep1(s, 'path.join(__dirname, "팜모니_발표자료_집v4.pptx")', 'path.join(__dirname, "팜모니_발표자료_집v5.pptx")')
open(PATH, "w", encoding="utf-8").write(s)
print("패치 완료. 본문 그림", order["그림"], "표", order["표"])
