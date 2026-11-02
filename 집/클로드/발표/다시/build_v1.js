// 팜모니 AI 발표자료 — 처음부터 다시 v1 (2026-11-02, 집 클로드)
// 근거: ./목차_v1_2026-11-02.md, ../부록재료_정정과_이용조건확인_2026-11-02.md, 정형·이미지 최종 검증 README, 카탈로그
// 실행: NODE_PATH=<pptxgenjs·jszip이 있는 node_modules> node build_v1.js
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");
const OUT = path.join(__dirname, "팜모니AI_발표자료_다시v1.pptx");

// ---- 디자인 토큰: 흰 바탕 + 짙은 글자, 미션별 색 하나씩, 핵심 숫자만 빨강 ----
const FONT = "Malgun Gothic";
const K = {
  ink: "1B2421", sub: "4A5652", gray: "7A8581", line: "D5DDD9", tint: "F2F5F4", white: "FFFFFF",
  dark: "163D2F", green: "1F6F50", greenT: "E6F1EC", blue: "2B5C8A", blueT: "E7EEF6", red: "C8333F", amber: "B7791F", amberT: "FBF3E4",
};
const FOOT = "팜모니 AI · 제1회 전국 농과계 대학 농업 AI 경진대회";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 × 7.5
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = "팜모니 AI 발표자료";

pres.defineSlideMaster({
  title: "BODY", background: { color: K.white },
  objects: [
    { line: { x: 0.6, y: 1.55, w: 12.13, h: 0, line: { color: K.line, width: 1 } } },
    { text: { text: FOOT, options: { x: 0.6, y: 7.0, w: 9, h: 0.3, fontFace: FONT, fontSize: 10, color: K.gray, margin: 0 } } },
  ],
  slideNumber: { x: 12.23, y: 7.0, w: 0.5, h: 0.3, fontFace: FONT, fontSize: 10, color: K.gray, align: "right" },
});
pres.defineSlideMaster({ title: "DARK", background: { color: K.dark }, objects: [] });

let n = 0;
const nm = (p) => `${p}_${++n}`;
const T = (s, text, o) => s.addText(text, Object.assign({ fontFace: FONT, fontSize: 16, color: K.ink, margin: 0, valign: "top", isTextBox: true, objectName: nm("txt") }, o));
const R = (s, x, y, w, h, fill, o = {}) => s.addShape(pres.shapes.RECTANGLE, Object.assign({ x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0.75 }, objectName: nm("box") }, o));
const RR = (s, x, y, w, h, fill, o = {}) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, Object.assign({ x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill, width: 0.75 }, objectName: nm("card") }, o));
const dot = (s, x, y, d, label, fill, fs = 14) => {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: nm("dot") });
  T(s, label, { x, y, w: d, h: d, align: "center", valign: "middle", fontSize: fs, bold: true, color: K.white });
};
const arrowR = (s, x, y, w = 0.32, h = 0.4, c = K.gray) => s.addShape(pres.shapes.RIGHT_ARROW, { x, y, w, h, fill: { color: c }, line: { color: c, width: 0 }, objectName: nm("arrow") });
// 글머리: 문자열 또는 [굵은 앞말, 뒷말]
const bl = (items, size = 16, color = K.ink, gap = 8) => items.map((it, i) => {
  const o = { bullet: { indent: 16 }, fontSize: size, color, paraSpaceAfter: gap, breakLine: i < items.length - 1 };
  return { text: Array.isArray(it) ? it[0] + it[1] : it, options: o };
});

// 본문 장: 섹션 표시(미션 색) + 제목(결론 문장)
function body(sec, title, color) {
  const s = pres.addSlide({ masterName: "BODY" });
  R(s, 0.6, 0.42, 0.12, 0.3, color);
  T(s, sec, { x: 0.82, y: 0.42, w: 8, h: 0.3, fontSize: 13, bold: true, color, valign: "middle" });
  T(s, title, { x: 0.6, y: 0.78, w: 12.13, h: 0.7, fontSize: 26, bold: true, color: K.ink, valign: "middle" });
  return s;
}
// 맺음 띠: 장의 결론 한 줄
function take(s, text, color, y = 6.2) {
  R(s, 0.6, y, 12.13, 0.6, color === K.blue ? K.blueT : color === K.amber ? K.amberT : K.greenT);
  R(s, 0.6, y, 0.08, 0.6, color);
  T(s, text, { x: 0.9, y, w: 11.7, h: 0.6, fontSize: 16, bold: true, color: K.ink, valign: "middle" });
}
function table(s, rows, o, head = K.dark) {
  const data = rows.map((r, i) => r.map((c) => ({
    text: String(c), options: i === 0 ? { bold: true, color: K.white, fill: { color: head } } : { color: K.ink, fill: { color: i % 2 ? K.white : K.tint } },
  })));
  s.addTable(data, Object.assign({ fontFace: FONT, fontSize: 13, border: { type: "solid", color: K.line, pt: 0.75 }, valign: "middle", margin: [4, 6, 4, 6], objectName: nm("table") }, o));
}
const AX = { catAxisLabelColor: K.sub, valAxisLabelColor: K.sub, catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT, titleFontFace: FONT, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  valGridLine: { color: "E3E8E6", size: 0.75 }, catGridLine: { style: "none" }, catAxisLabelRotate: 0, titleColor: K.sub, titleFontSize: 13 };

// ================= 표지 =================
{
  const s = pres.addSlide({ masterName: "DARK" });
  R(s, 0.8, 2.2, 0.1, 2.2, K.red);
  T(s, "팜모니 AI", { x: 1.15, y: 2.1, w: 11, h: 0.6, fontSize: 20, bold: true, color: "A9C9B9" });
  T(s, "현장에서 그 시각에 알 수 있는 정보로\n예측하고 판별한다", { x: 1.15, y: 2.7, w: 11, h: 1.5, fontSize: 36, bold: true, color: K.white });
  T(s, "정형(배지 온도·EC 예측) · 이미지(딸기 사진 진위 판별)", { x: 1.15, y: 4.25, w: 11, h: 0.4, fontSize: 16, color: "C9DCD2" });
  T(s, "발표: 조나단   |   팀원: 조나단 · 박찬유 · 반지원 · 김태경 · 김우진", { x: 1.15, y: 5.6, w: 11, h: 0.35, fontSize: 13, color: "C9DCD2" });
  T(s, "2026. 11. 6  ·  대구 EXCO", { x: 1.15, y: 6.0, w: 11, h: 0.35, fontSize: 13, color: "8FB3A2" });
  s.addNotes("[말할 것] 팜모니 AI입니다. 정형과 이미지 두 미션을 하나의 원칙으로 풀었습니다: 현장에서 그 시각에 알 수 있는 정보만으로 예측·판별하도록 만들고, 그것을 검증으로 확인하는 것입니다.\n\n[노트] 발표 내용은 최종 검증에 제출한 코드·모델(정형 팜모니_정형.zip, 이미지 팜모니_이미지.zip)을 기준으로 합니다. 실제로 한 것과 향후 계획은 마지막 장에서 구분합니다.");
}

// ================= 1. 한 장 요약 =================
{
  const s = body("요약", "두 미션 모두 '현장에서 알 수 있는 정보'만으로 풀었다", K.dark);
  const card = (x, color, tintC, head, sub, stats) => {
    RR(s, x, 1.85, 5.95, 3.55, tintC);
    T(s, head, { x: x + 0.35, y: 2.05, w: 5.3, h: 0.45, fontSize: 20, bold: true, color });
    T(s, sub, { x: x + 0.35, y: 2.5, w: 5.3, h: 0.35, fontSize: 13, color: K.sub });
    stats.forEach((st, i) => {
      T(s, st[0], { x: x + 0.35 + i * 2.8, y: 3.05, w: 2.7, h: 1.0, fontSize: 44, bold: true, color: i === 0 ? K.red : color });
      T(s, st[1], { x: x + 0.35 + i * 2.8, y: 4.1, w: 2.6, h: 1.0, fontSize: 13, color: K.ink });
    });
  };
  card(0.6, K.green, K.greenT, "정형 미션", "두 온실의 시간별 배지 온도·EC 예측 · 최종 답안 24회차 구성", [[".4608", "온도 RMSE(℃)\n공개 점수, 낮을수록 좋음"], [".1148", "EC RMSE\n공개 점수"]]);
  card(6.78, K.blue, K.blueT, "이미지 미션", "딸기 시설재배 사진의 실제·가짜 판별 · 최종 나단11", [[".97", "Accuracy\n운영진 채점(R21)"], ["25.59", "환산 점수\n(30점 만점)"]]);
  T(s, bl([
    "정형: 하루 단위 오차의 답은 '어제'에 있다 → 같은 동의 어제를 잇는 구조로, 대회 자료 표식은 빼고 다시 만들었다",
    "이미지: 공개 데이터로 가짜를 직접 만들어 학습·검증했다 → 같은 절차를 현장 사진으로 다시 돌릴 수 있다",
  ], 15), { x: 0.6, y: 5.6, w: 12.13, h: 1.2 });
  s.addNotes("[말할 것] 결과부터 말씀드립니다. 정형은 온도 .4608, EC .1148로 두 목표 모두 우리 최고 기록인 24회차가 최종입니다. 이미지는 운영진 채점 Accuracy .97입니다. 두 미션의 공통 원칙은 현장에서 그 시각에 알 수 있는 정보만 쓰는 것입니다.\n\n[노트]\n- 정형 점수 출처: 플랫폼 공개 리더보드(24회차, submission_35e.csv, 10-16 18:22 업로드). 정형 README ⑤·⑦, 제출/성능_정리.md.\n- 이미지 점수 출처: 운영진 R21 채점(10-16 16:00), 팜모니_나단11.zip(SHA256 6fe04909…27c1). 이미지 README 0절.\n- 공개 점수는 대회 운영진의 비공개 평가 자료로 낸 값이며, 우리는 평가 정답에 접근하지 않았습니다.");
}

// ================= 정형 구분 =================
function divider(label, title, sub, color) {
  const s = pres.addSlide({ masterName: "DARK" });
  R(s, 0, 0, 0.35, 7.5, color);
  T(s, label, { x: 1.1, y: 2.5, w: 10, h: 0.5, fontSize: 18, bold: true, color: "A9C9B9" });
  T(s, title, { x: 1.1, y: 3.0, w: 11, h: 0.9, fontSize: 40, bold: true, color: K.white });
  T(s, sub, { x: 1.1, y: 4.0, w: 11, h: 0.5, fontSize: 16, color: "C9DCD2" });
  return s;
}
divider("PART 1", "정형 미션", "두 온실(F13·F47)의 시간별 배지 온도·EC 예측", K.green).addNotes("[말할 것] 먼저 정형 미션입니다. 약 5분 동안 문제, 핵심 발견, 모델, 현장형 정리, 검증, 현장 해석 순서로 말씀드립니다.");

// ================= 2. 문제와 자료 =================
{
  const s = body("정형 · 문제와 자료", "하루 기록이 섞여 있고, 같은 동이 두 온실 ID를 오간다", K.green);
  T(s, "문제", { x: 0.6, y: 1.85, w: 5.6, h: 0.4, fontSize: 18, bold: true, color: K.green });
  T(s, bl([
    "F13·F47 두 온실의 시간별 배지 온도와 EC 예측",
    "입력: 그 시각까지의 실내·외기·구동기 기록",
    "학습 400일(9,600행), 평가 60기록 — 평가는 모두 2차 구간",
    "평가 행은 같은 온실의 현재·이전 입력만 쓸 수 있음",
  ], 16), { x: 0.6, y: 2.35, w: 5.7, h: 3.0 });
  T(s, "자료를 들여다보고 찾은 것", { x: 6.9, y: 1.85, w: 5.8, h: 0.4, fontSize: 18, bold: true, color: K.green });
  [["1", "하루(24행) 단위 기록이 번호 순서와 상관없이 섞여 있다"], ["2", "같은 동이 F13과 F47 두 ID를 오간다 — 확실한 짝의 약 1/4"], ["3", "기록 번호는 문제설명서의 '온실별 상대 일차'다"]].forEach((v, i) => {
    RR(s, 6.9, 2.35 + i * 1.0, 5.83, 0.85, K.tint);
    dot(s, 7.1, 2.52 + i * 1.0, 0.5, v[0], K.green);
    T(s, v[1], { x: 7.8, y: 2.35 + i * 1.0, w: 4.8, h: 0.85, fontSize: 15, valign: "middle" });
  });
  take(s, "그래서 '어제 기록'을 찾아 이을 수 있느냐가 정형 성능의 열쇠가 된다", K.green);
  s.addNotes("[말할 것] 과제는 두 온실의 시간별 배지 온도와 EC 예측입니다. 자료를 보니 하루 단위 기록이 섞여 있고, 같은 동이 두 온실 ID를 오가고 있었습니다. 그래서 같은 동의 어제를 찾아 잇는 것이 핵심이 됐습니다.\n\n[노트]\n- 기록 = 한 온실의 하루 24행(F13_185처럼 온실 ID + 기록 번호). 기록 번호는 문제설명서의 '온실별 상대 일차'(정형 README 용어표).\n- 1차·2차 구간 = 기록 번호 179 미만·이상. 평가 60기록은 모두 2차 구간.\n- 같은 동이 두 ID를 오가는 것: 외기·실내 이어짐으로 확인한 확실한 짝 중 약 1/4이 다른 온실 ID로 이어짐(데이터 단서 카탈로그).\n- 규정: 평가 행 특징은 같은 온실의 현재·이전 입력만, 학습 행 특징은 평가 입력이 빈 것처럼(MASK) 계산.");
}

// ================= 3. 핵심 발견 =================
{
  const s = body("정형 · 핵심 발견", "하루 통째로 어긋나는 오차는 '오늘 입력'이 아니라 '어제'에 답이 있다", K.green);
  const st = [[".49", "오차 중 '하루 수준' 몫", "하루 평균만큼 통째로 어긋나는 몫이\n오차의 절반 가량"], ["−.64", "그날 입력으로 설명되는 정도(R²)", "음수 = 그날 입력 요약으로는\n전혀 설명되지 않음"], [".64", "어제 오차와의 상관", "연속한 날의 하루 평균 오차가\n강하게 이어짐"]];
  st.forEach((v, i) => {
    const x = 0.6 + i * 4.1;
    RR(s, x, 1.9, 3.9, 3.6, K.tint);
    T(s, v[0], { x: x + 0.3, y: 2.1, w: 3.4, h: 1.1, fontSize: 54, bold: true, color: i === 2 ? K.red : K.green });
    T(s, v[1], { x: x + 0.3, y: 3.3, w: 3.4, h: 0.45, fontSize: 16, bold: true });
    T(s, v[2], { x: x + 0.3, y: 3.85, w: 3.4, h: 1.2, fontSize: 14, color: K.sub });
  });
  T(s, "학습 자료의 다른 온실들에서 잰 중앙값(같은 기준 모델의 교차검증 오차, 카탈로그 6.997)", { x: 0.6, y: 5.65, w: 12, h: 0.35, fontSize: 12, color: K.gray });
  take(s, "처방: 같은 동의 진짜 '어제'를 찾아, 어제의 실측과 예측을 오늘로 넘긴다", K.green);
  s.addNotes("[말할 것] 오차를 뜯어보니 절반 가량이 '그날 하루가 통째로 어긋나는' 몫이었습니다. 이 몫은 그날 입력으로는 설명되지 않았고, 대신 어제의 오차와 강하게 이어져 있었습니다. 우리 두 온실만의 일이 아니라 다른 온실에서도 같았습니다. 그래서 처방은 어제를 잇는 것입니다.\n\n[노트]\n- 분석 DLV1(카탈로그 6.997): 학습 자료의 온실들에 같은 기준 모델(SELF_LR, 5덩어리 교차검증 예측)을 적용해 오차를 분해.\n- 하루 수준 몫: 다른 온실 중앙 .49(F13 .49, F47 .53). 그날 입력 요약으로 하루 평균 오차를 설명하는 교차 R²: 다른 온실 중앙 −.64(F13 −.18, F47 −.12, 모두 음수). 연속한 날 하루 평균 오차 상관: 다른 온실 중앙 .64.\n- F13·F47은 기록 번호 이웃이 진짜 어제가 아니어서(기록이 섞임) 번호 이웃 상관은 .15·.29로 낮음 → 진짜 어제를 찾아야 하는 이유.\n- 서술용 분석이며 채택 판정이 아닙니다.");
}

// ================= 4. 모델 3단계 =================
{
  const s = body("정형 · 모델", "기본 예측 위에 '어제'를 얹고, 추운 시간대를 보정한다", K.green);
  const steps = [
    ["1", "기본 예측", ["세 모델을 섞음: 물리식+부스팅(BASE), 잔차 모델(CODEX), TabPFN", "실내 8℃ 이하에서는 TabPFN을 빼는 추위 게이트"]],
    ["2", "어제 이어 붙이기", ["링커가 같은 동의 '어제 기록'을 고름(0시 입력만 사용)", "어제 23시 실측이 있으면 앵커, 없으면 어제 예측으로 사슬"]],
    ["3", "추운 행 보정", ["EC가 높은 날의 추운 시간대 배지 온도를 소폭 올림(TX3)", "EC도 같은 구조: 기본 예측 + 어제 EC 앵커"]],
  ];
  steps.forEach((v, i) => {
    const x = 0.6 + i * 4.15;
    RR(s, x, 1.9, 3.85, 3.9, i === 1 ? K.greenT : K.tint);
    dot(s, x + 0.3, 2.1, 0.62, v[0], i === 1 ? K.red : K.green, 18);
    T(s, v[1], { x: x + 1.05, y: 2.1, w: 2.7, h: 0.62, fontSize: 20, bold: true, valign: "middle" });
    T(s, bl(v[2], 15, K.ink, 10), { x: x + 0.3, y: 3.0, w: 3.35, h: 2.7 });
    if (i < 2) arrowR(s, x + 3.88, 3.6, 0.24, 0.42);
  });
  take(s, "섞는 비율·문턱은 실행 전에 정한 검증 기준으로 골랐다(TabPFN 0.4는 리더보드 비교 — 부록 A7)", K.green);
  s.addNotes("[말할 것] 모델은 세 단계입니다. 기본 예측은 세 모델을 섞고, 추운 시간대에는 TabPFN을 뺍니다. 그 위에 링커가 찾은 같은 동의 어제를 얹습니다. 어제 실측이 있으면 그 값으로, 없으면 어제의 예측으로 이어 갑니다. 마지막으로 추운 시간대를 조금 보정합니다.\n\n[노트]\n- 섞는 식: g = clip((실내온도−8)/2, 0, 1), pred = 0.4·BASE + (0.2+0.4(1−g))·CODEX + 0.4·g·TabPFN(정형 README ② 온도 1).\n- FL 링커: 평가 기록마다 0시에 독립적으로 같은 동의 어제를 고름. 특징은 후보 23시→대상 0시의 외기·실내·구동기 이어짐 등, 가중치는 학습 정답의 이어지는 짝으로만 맞춤(README ② 온도 2).\n- 어제 앵커: pred_h = q_h + β_h·clip(A − 0.36 − q_0, −2, 2), A = 어제 23시 실측, β_h = 시각별 감쇠(0시 .76 → 9시 .13, 이후 .16~.24). 조건: 링커 확신 p ≥ .8 등(README ② 온도 3).\n- 예측 사슬: 어제에 실측이 없을 때 어제의 최종 23시 예측을 0.9·exp(−h/8)로 줄여 이어 붙임(README ② 온도 5).\n- TX3: +0.25℃·clip((cmEC−0.8)/0.4)·clip((10−실내온도)/2)(README ② 온도 4).\n- EC: R3 + TabPFN 구성원 + 하루 수준 보정(HR5) + 어제 EC 앵커(README ② EC).");
}

// ================= 5. 현장형으로 다시 만들기 =================
{
  const s = body("정형 · 현장형 정리", "대회 자료에만 있는 표식은 빼고, 현장에도 있는 '어제'만 남겼다", K.green);
  RR(s, 0.6, 1.9, 5.95, 3.95, K.tint);
  T(s, "뺀 것 — 대회 자료에만 있는 표식", { x: 0.9, y: 2.05, w: 5.4, h: 0.45, fontSize: 18, bold: true, color: K.sub });
  T(s, bl([
    "기록 번호 자체를 특징으로 쓰기(day)",
    "2차 구간 표시(번호 ≥ 179, second)",
    "결번·평가 기록 유무로 갈리는 규칙(W2)",
    "EC의 대회 구조 규칙(SG2·B5)",
  ], 16, K.ink, 10), { x: 0.9, y: 2.6, w: 5.4, h: 3.0 });
  RR(s, 6.78, 1.9, 5.95, 3.95, K.greenT);
  T(s, "남긴 것 — 현장에도 있는 정보", { x: 7.08, y: 2.05, w: 5.4, h: 0.45, fontSize: 18, bold: true, color: K.green });
  T(s, bl([
    "같은 동의 어제: 대회에서는 링커로 찾고, 현장에서는 달력으로 정해짐",
    "앞 일차 기록의 이력: 기록 번호 = 공식 '온실별 상대 일차'라 현장과 같은 순서",
    "어제 23시 실측: 현장 센서에서 그대로 얻는 값",
  ], 16, K.ink, 10), { x: 7.08, y: 2.6, w: 5.4, h: 3.0 });
  take(s, "남은 한계: 계절 지표의 2차 구간 날짜 복원은 대회 자료 성질에 기댄다 — 현장에서는 달력 날짜로 대체", K.amber);
  s.addNotes("[말할 것] 마지막 단계에서 모델을 현장형으로 정리했습니다. 대회 자료에만 있는 표식, 예를 들어 기록 번호 자체나 2차 구간 표시, 평가 기록이 있느냐로 갈리는 규칙은 뺐습니다. 남긴 것은 현장에도 있는 정보, 즉 어제입니다. 대회에서는 어제를 링커로 찾아야 했지만 현장에서는 달력으로 정해집니다. 남은 한계는 계절 지표 하나입니다.\n\n[노트]\n- day·second 제거: 정형 README ② 온도 1, ⑤ 표(검증기 4종 혼재: DIAG10 +0.4%, EL1 −0.5%, EXT10 −0.4%, EXT12 −0.6%) → 현장형 구성(submission_27, 17회차 온도 .4817, 당시 최고)으로 제출 비교로 채택.\n- W2(row_id 결번·앞 일차 기록의 평가 자료 존재 여부 규칙)는 최종 답안에 없음(README ④, 주최 측 10/16 답변 1항).\n- EC SG2·W2·B5 제거: README ⑤ 표, 현장형 구성(submission_25·27).\n- 기록 번호 = 문제설명서의 '온실별 상대 일차'(README 용어표) → BASE의 앞 일차 이력은 번호 순서로 잇되, 현장에서도 같은 순서.\n- 계절 지표: EC/season.py가 '2차 구간 외기가 1차 구간 어떤 날의 외기와 거의 같다'는 대회 자료 성질로 기록 번호→계절 대응을 만듦(README ② EC 계절 지표). 현장에서는 달력 날짜가 직접 주어지므로 필요 없는 단계. 계절 지표를 뺀 성능은 미실시.\n- 참고: BASE 이력을 번호 순 대신 링커의 어제로 다시 잇는 판(20회차)은 검증에서는 좋았지만 리더보드에서 나빠져 최종에 넣지 않았습니다(6쪽).");
}

// ================= 6. 검증과 개선 =================
{
  const s = body("정형 · 검증과 개선", "실행 전에 고정한 기준을 통과한 변경만 리더보드에서도 좋아졌다", K.green);
  T(s, "검증 방법", { x: 0.6, y: 1.85, w: 4.8, h: 0.4, fontSize: 18, bold: true, color: K.green });
  T(s, bl([
    "학습 기록 일부를 가려서 만든 검증기 여러 개(추운 날을 뺀 EXT10·EXT12, 무작위 DIAG10 등)",
    "실행 전에 기준을 코드에 고정: 모든 시드·검증기에서 같은 방향 + 블록 부트스트랩 P < .025",
    "기준을 못 넘으면 '가설 검정 제출'로만 확인",
  ], 15, K.ink, 10), { x: 0.6, y: 2.35, w: 4.9, h: 3.6 });
  table(s, [
    ["회차", "바꾼 것", "사전 기준", "온도 RMSE"],
    ["17", "현장형 모델(표식 제거 + 어제 앵커)", "제출 비교", ".4817"],
    ["19", "+ 예측 사슬", "통과", ".4706"],
    ["20", "BASE 이력을 링커 어제로 다시 잇기", "근소 미달", ".4754 ▲"],
    ["21", "사슬 확신 문턱 .8 → .5", "통과", ".4685"],
    ["22", "+ 범위 밖 보정(OOD1·TX3L)", "미달", ".5000 ▲"],
    ["24", "최종: 일치 앵커·사슬 확장(온도·EC)", "제출 비교", ".4608"],
  ], { x: 5.75, y: 1.9, w: 6.98, colW: [0.75, 3.5, 1.3, 1.43], fontSize: 13, rowH: 0.52 });
  take(s, "통과한 변경(19·21회차)은 그대로 옮겨졌고, 근소하게 못 넘은 변경(20·22회차)은 크게 어긋났다", K.green);
  s.addNotes("[말할 것] 검증은 학습 기록을 가려 만든 여러 검증기로 했고, 채택 기준은 실행 전에 코드에 고정했습니다. 결과가 흥미로웠습니다. 기준을 통과한 변경은 리더보드에서도 좋아졌고, 근소하게 못 넘은 변경은 크게 나빠졌습니다. 그래서 기준을 지키는 것 자체가 실패를 거르는 장치였습니다.\n\n[노트]\n- 점수 출처: 플랫폼 공개 리더보드(제출/성능_정리.md 정형 표). ▲ = 직전 최고보다 나빠짐.\n- 17회차(submission_27): 온도 .4817, EC .1170. 19회차 .4706, 20회차 .4754, 21회차 .4685, 22회차 .5000, 23회차 .4665, 24회차 .4608·EC .1148(최종).\n- 20회차: 검증기 12/12셀 개선 방향, 새 배치 15/16셀(P .013)이었지만 전 셀 조건 1셀 미달 → 가설 검정 제출 → 리더보드 악화(카탈로그 7.037~7.039).\n- 22회차: 검증 11/12(P .106), 추운 영역을 바꾸는 변경 → 리더보드 크게 악화.\n- 리더보드 점수 사용처는 자기 점검 4번(부록 A1): 제출물 비교·선택, 상수 결정은 TabPFN 비중 0.4 한 건(부록 A7).\n- 검증기들은 F13·F47 학습 기록을 가려 만든 것이며 평가 자료가 아닙니다. 출처표는 부록 A4.");
}

// ================= 7. 현장 해석 =================
{
  const s = body("정형 · 농업적 해석과 현장 조건", "'어제' 정보를 끄면 오차가 온도 3~10%, EC 8~31% 늘어난다", K.green);
  const mk = (x, title, cats, vals, color) => s.addChart(pres.charts.BAR, [{ name: title, labels: cats, values: vals }], Object.assign({
    x, y: 1.85, w: 4.0, h: 3.9, barDir: "col", chartColors: [color], showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelColor: K.ink, dataLabelFontSize: 12,
    valAxisMinVal: 0, valAxisMaxVal: 35, valAxisMajorUnit: 10, showTitle: true, title, objectName: nm("chart"),
  }, AX));
  mk(0.6, "온도: 오차 증가(%)", ["EXT10", "EXT12", "DIAG10"], [10.4, 3.1, 3.4], K.green);
  mk(4.75, "EC: 오차 증가(%)", ["DIAG10", "EL1", "P2LOO"], [8.3, 30.5, 11.0], K.green);
  RR(s, 9.0, 1.9, 3.73, 3.85, K.tint);
  T(s, "현장에 주는 뜻", { x: 9.25, y: 2.05, w: 3.3, h: 0.4, fontSize: 17, bold: true, color: K.green });
  T(s, bl([
    "배지 상태는 하루 단위로 이어진다 — 어제가 오늘의 가장 좋은 단서",
    "센서 기록이 끊기지 않아야 한다(어제 23시 실측이 있으면 가장 좋음)",
    "추운 날·EC 높은 날에 오차가 크다",
  ], 14, K.ink, 8), { x: 9.25, y: 2.55, w: 3.35, h: 3.1 });
  take(s, "기본 예측이 성능의 대부분을 내고, '어제'는 그 위에 의미 있는 개선을 더한다", K.green);
  s.addNotes("[말할 것] 어제 정보를 모두 끄고 일반 방법만 남기면, 학습 자료 검증기에서 오차가 온도 3~10%, EC 8~31% 늘어납니다. 현장 해석은 간단합니다. 배지 상태는 하루 단위로 이어지므로, 센서 기록이 끊기지 않고 어제 실측이 남아 있는 것이 정확도의 조건입니다.\n\n[노트]\n- ABL1 끄기 실험(11-02, 서술용, 계획 사전 고정 커밋 45ac69c5): 일반 방법만(온도 NX+TX3, EC pre+HR5) vs 최종 구성, 2차 행 RMSE.\n  - 온도: EXT10 .8387→.7599(+10.4%), EXT12 .9059→.8791(+3.1%), DIAG10 .7388→.7146(+3.4%).\n  - EC: DIAG10 .2522→.2329(+8.3%), EL1 .2787→.2135(+30.5%), P2LOO .1913→.1723(+11.0%).\n  - 블록 부트스트랩 '끈 판이 나쁘지 않을 확률': 온도 2차 .025, EC 2차 .034.\n- 근거 파일: 집/클로드/발표/ABL1_어제이어붙이기_끄기_결과_2026-11-02.md, 카탈로그 7.069.\n- 검증기는 학습 기록을 가려 만든 것이며 평가 자료가 아닙니다.\n- 한계: 현장형 사슬(실제 달력 어제)로 다시 잰 성능은 미실시.");
}

// ================= 이미지 구분 =================
divider("PART 2", "이미지 미션", "딸기 시설재배 사진이 실제인지, 생성·편집된 가짜인지 판별", K.blue).addNotes("[말할 것] 다음은 이미지 미션입니다. 약 3분 30초 동안 전략, 모델, 검증, 현장 적용을 말씀드립니다.");

// ================= 8. 문제와 전략 =================
{
  const s = body("이미지 · 문제와 전략", "공개 데이터의 실제 사진으로 가짜를 직접 만들어 학습했다", K.blue);
  const fl = [
    ["실제 사진 확보", "AI Hub 「지능형 수직농장\n통합 데이터(딸기)」\n원천 152,043장"],
    ["가짜 직접 제작", "필터·합성 · SD1.5·SDXL\n생성·부분 생성·스타일\nQwen·ChatGPT 편집"],
    ["배포 형식 맞춤", "768×1024 크롭\nJPEG 품질 92 · 4:2:0\n(공개 샘플 실측)"],
    ["학습·검증", "촬영 묶음 단위\n교차검증 + 따로 떼어 둔\n봉인 검증 묶음"],
  ];
  fl.forEach((v, i) => {
    const x = 0.6 + i * 3.1;
    RR(s, x, 1.95, 2.8, 3.3, i === 1 ? K.blueT : K.tint);
    dot(s, x + 0.25, 2.15, 0.55, String(i + 1), K.blue, 16);
    T(s, v[0], { x: x + 0.95, y: 2.15, w: 1.8, h: 0.55, fontSize: 17, bold: true, valign: "middle" });
    T(s, v[1], { x: x + 0.25, y: 2.95, w: 2.4, h: 2.1, fontSize: 14, color: K.sub });
    if (i < 3) arrowR(s, x + 2.83, 3.35, 0.24, 0.4);
  });
  T(s, "대회가 주는 학습 사진 없이, 정답(실제/가짜)을 아는 자료를 우리가 만들었다", { x: 0.6, y: 5.45, w: 12, h: 0.4, fontSize: 15, color: K.sub });
  take(s, "같은 절차는 현장 사진만 있으면 다시 돌릴 수 있다 — 학습도, 검증도", K.blue);
  s.addNotes("[말할 것] 이미지 미션은 정답이 있는 학습 사진이 따로 없었습니다. 그래서 공개 데이터의 실제 딸기 사진을 받아, 여러 방법으로 가짜를 직접 만들고, 대회 배포 형식으로 맞춰 학습했습니다. 검증도 우리가 만든 자료로 했습니다. 이 절차는 현장 사진만 있으면 그대로 다시 돌릴 수 있습니다.\n\n[노트]\n- 실제 사진: AI Hub 596, 원천 tar 306개 152,043장(팀원 계정 이용 신청·승인). 대회 공개 샘플도 같은 데이터셋에서 나온 것으로 보임(공개 가짜 일부의 원본을 지문 DB에서 찾음, README_data 4절).\n- 가짜 제작: 프로그램 필터·합성, SD1.5, SDXL 전체 생성·전체 스타일·부분 스타일·부분 생성, Qwen-Image-Edit(연구실 GPU 서버에서 직접 실행), ChatGPT 이미지 편집 600장(AI Hub 사진 525장 입력). README ④, data/gpt_generation.\n- 기본 판별기 학습: AI Hub 실제 2,467 + 자체 가짜 7,124(README ④).\n- 교차검증: 원본 촬영 묶음·농장 단위로 나눠, 같은 원본의 실제와 가짜가 다른 폴드에 갈리지 않게 함.\n- 한계: AI Hub 44농장을 모두 학습에 써서, 처음 보는 온실 사진의 성능은 미실시(카탈로그 7.070).");
}

// ================= 9. 모델 구성 =================
{
  const s = body("이미지 · 모델", "핵심은 사진 한 장만 보는 일반 판별기, 대회용 장치는 따로 표시했다", K.blue);
  RR(s, 0.6, 1.9, 3.7, 3.95, K.blueT);
  T(s, "① 기본 판별", { x: 0.85, y: 2.05, w: 3.3, h: 0.45, fontSize: 18, bold: true, color: K.blue });
  T(s, bl(["CLIP 의미 특징 + 로지스틱", "저수준 통계 38개(잡음·선명도·JPEG 격자) + 부스팅", "두 확률의 평균"], 14, K.ink, 8), { x: 0.85, y: 2.6, w: 3.3, h: 3.0 });
  arrowR(s, 4.38, 3.6, 0.28, 0.42);
  RR(s, 4.75, 1.9, 3.7, 3.95, K.blueT);
  T(s, "② 유형별 전문가", { x: 5.0, y: 2.05, w: 3.3, h: 0.45, fontSize: 18, bold: true, color: K.blue });
  T(s, bl(["기본이 놓치기 쉬운 유형만 따로 봄", "부분 생성·부분 스타일·국소 조각(형식 무관)", "하나라도 문턱을 넘으면 가짜 쪽으로"], 14, K.ink, 8), { x: 5.0, y: 2.6, w: 3.3, h: 3.0 });
  RR(s, 8.9, 1.9, 3.83, 3.95, K.amberT, { line: { color: K.amber, width: 1.25, dashType: "dash" } });
  T(s, "대회 자료용 장치 (현장에서는 꺼짐)", { x: 9.1, y: 2.05, w: 3.5, h: 0.45, fontSize: 16, bold: true, color: K.amber });
  T(s, bl([
    "지문 DB 원본 대조: AI Hub에 원본이 있는 사진에서만 작동",
    "형식 게이트: JPEG 품질 92·4:2:0 파일에서만 전문가 4종(필터·조각·GPT 재그리기·GPT 편집)을 켬",
  ], 14, K.ink, 8), { x: 9.1, y: 2.6, w: 3.45, h: 3.1 });
  take(s, "모든 계산은 사진 한 장 단위로 독립 · 문턱은 학습 자료에서 정함 · 판정 임계값 0.5", K.blue);
  s.addNotes("[말할 것] 핵심은 사진 한 장만 보고 판단하는 기본 판별기와 유형별 전문가입니다. 여기에 대회 자료에서만 작동하는 장치 두 개가 붙어 있습니다. 하나는 AI Hub 원본과 대조하는 지문 DB이고, 다른 하나는 대회 파일 형식일 때만 일부 전문가를 켜는 형식 게이트입니다. 이 둘은 현장 사진에서는 거의 작동하지 않으므로 따로 표시했습니다.\n\n[노트]\n- 계산 순서(README ⑤): 1) p = (p_CLIP + p_low)/2 → 2) 12×16 색 지문 R1 규칙 → 3) 전문가 E_pg·E_ps·E_L → 4) 정밀 원본 대조(가짜 확정 p≥.95 / 진짜 확정 rc p≤.3) → 5) 형식 게이트 → 6) 전문가 H(E_f·E_ps2·E_g7)·E_gr → 7) E_gpt(게이트 안, rc 아닐 때).\n- 전문가 문턱 θ: 학습 OOF·실제 풀에서 실제 사진을 0.25~0.4%만 뒤집는 분위(README ⑤).\n- 지문 DB: fp_db(사진별 12×16 RGB 평균), finefp_db(흑백 24×32 + 라플라시안 에너지 192값), 152,043장, 사진·파일명 없음. 상세는 부록 A6·A8.\n- 형식 게이트: PIL로 연 파일이 JPEG, 4:2:0, PIL quality=92 양자화표와 완전히 같을 때만 참. 게이트 밖이면 전문가 H·E_gr·E_gpt 미적용.");
}

// ================= 10. 검증과 개선 =================
{
  const s = body("이미지 · 검증과 개선", "학습에 쓰지 않은 검증 묶음에서 균형 정확도 .94~.96", K.blue);
  s.addChart(pres.charts.BAR, [{ name: "균형 정확도", labels: ["R3", "R4", "R5"], values: [96.2, 94.7, 94.4] }], Object.assign({
    x: 0.6, y: 1.85, w: 5.4, h: 3.9, barDir: "col", chartColors: [K.blue], showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelColor: K.ink, dataLabelFontSize: 13,
    valAxisMinVal: 85, valAxisMaxVal: 100, valAxisMajorUnit: 5, showTitle: true, title: "봉인 검증 묶음 균형 정확도(%) · 제출 모델", objectName: nm("chart"),
  }, AX));
  T(s, "검증 방법", { x: 6.4, y: 1.85, w: 6.3, h: 0.4, fontSize: 18, bold: true, color: K.blue });
  T(s, bl([
    "봉인 검증 묶음 R2~R5: 사진은 학습·문턱 계산에 쓰지 않고, 후보 채택 판정 때만 열었다",
    "대표값은 원본 겹침이 없는 R3~R5(R2는 학습 가짜와 같은 원본이 섞임 — 부록 A4)",
    "운영진 채점 Accuracy .97(공개 21장은 모든 후보에서 21/21 유지 점검)",
  ], 14, K.ink, 8), { x: 6.4, y: 2.3, w: 6.3, h: 2.2 });
  T(s, "실패에서 고친 것", { x: 6.4, y: 4.45, w: 6.3, h: 0.4, fontSize: 18, bold: true, color: K.blue });
  T(s, bl([
    "GPT 편집 전문가 첫 판: 놓친 26장 중 14장을 잡았지만 실제 오탐 기준을 근소하게 넘어 불통과",
    "원인 분해 → 원본과 거의 같은 '진짜 확정' 사진에서 오판 → 그 사진에는 적용하지 않게 수정",
  ], 14, K.ink, 8), { x: 6.4, y: 4.9, w: 6.3, h: 1.3 });
  take(s, "주의: 봉인 묶음도 학습과 같은 농장·같은 데이터셋 — 처음 보는 온실 성능은 미실시", K.amber);
  s.addNotes("[말할 것] 검증은 따로 떼어 둔 봉인 묶음으로 했습니다. 원본 겹침이 없는 세 묶음에서 균형 정확도는 94~96%였고, 운영진 채점은 97%였습니다. 실패에서 고친 사례로, GPT 편집 전문가 첫 판이 실제 사진 오탐 기준을 근소하게 넘었을 때 원인을 분해해 수정했습니다. 다만 봉인 묶음도 같은 농장 사진이라, 처음 보는 온실 성능은 확인하지 못했습니다.\n\n[노트]\n- 균형 정확도(나단11, 0.5 기준): R2 .9637, R3 .9617, R4 .9468, R5 .9436(10-20 끄기 실험 판 A, logs/ablation_원본대조_2026-10-20/요약표.md). R3 실제 429/가짜 471, R4 355/345, R5 356/344.\n- 겹침(매니페스트 대조, 11-02): R2~R5 농장·촬영 묶음 모두 학습과 겹침. 같은 원본은 R2만 겹침(실제 1,862장 중 1,090장의 원본으로 만든 SDXL 가짜가 기본 판별기 학습에). 최종 검증 README_data의 '촬영 묶음 안 겹치게' 문구는 사실과 달라 부록 A4에서 정정.\n- 봉인 묶음은 후보 채택 판정에 썼으므로 모델 선택에 쓴 자료입니다(공통 2).\n- E_gpt rc 제외: 첫 판은 빼기 검증에서 26장 중 14장을 새로 잡았지만 합집합 실제 오탐 +0.12%p, R5 −0.28%p로 불통과. 새 오판 대부분이 진짜 확정(rc) 사진이고 GPT 가짜 600장 중 rc는 0장 → rc 제외. 같은 자료로 다시 집계하면 기준 충족이지만, 불통과를 본 뒤 정한 변형이라 낙관적일 수 있음(README ⑤, IM_Egpt_rc제외_v1.md). 리더보드는 쓰지 않음.\n- 운영진 채점: R21(10-16 16:00) Accuracy .97, 환산 25.59.");
}

// ================= 11. 현장 적용 =================
{
  const s = body("이미지 · 현장 적용 조건과 한계", "대회용 장치를 끄고 다시 재 봤다", K.blue);
  s.addChart(pres.charts.BAR, [
    { name: "제출 모델", labels: ["R3", "R4", "R5"], values: [96.2, 94.7, 94.4] },
    { name: "지문 DB 끔", labels: ["R3", "R4", "R5"], values: [94.8, 92.4, 92.1] },
  ], Object.assign({
    x: 0.6, y: 1.85, w: 6.0, h: 3.9, barDir: "col", barGrouping: "clustered", chartColors: [K.blue, "9DB5CF"], showLegend: true, legendPos: "b", legendFontFace: FONT, legendFontSize: 12, legendColor: K.sub,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelColor: K.ink, dataLabelFontSize: 11,
    valAxisMinVal: 85, valAxisMaxVal: 100, valAxisMajorUnit: 5, showTitle: true, title: "균형 정확도(%) · 봉인 검증 묶음", objectName: nm("chart"),
  }, AX));
  RR(s, 6.95, 1.9, 5.78, 1.35, K.amberT);
  T(s, "형식 게이트까지 끈 성능", { x: 7.2, y: 2.0, w: 5.3, h: 0.4, fontSize: 16, bold: true, color: K.amber });
  T(s, "판정 진행 중 — 결과 반영 예정(ABL2, 계획 사전 고정)", { x: 7.2, y: 2.45, w: 5.3, h: 0.7, fontSize: 14, color: K.ink });
  T(s, "현장에서 바뀌는 것", { x: 6.95, y: 3.4, w: 5.8, h: 0.4, fontSize: 17, bold: true, color: K.blue });
  T(s, bl([
    "지문 DB 끄면 −1.4~−2.3%p, 주로 실제 사진 오탐 증가",
    "선명화한 실제 사진 90.7%, WebP 37.5%를 가짜로 판정 — 휴대폰 사진에 약할 수 있음",
    "처음 보는 온실 성능: 미실시",
  ], 14, K.ink, 8), { x: 6.95, y: 3.85, w: 5.8, h: 2.2 });
  take(s, "현장에서는 그 농장 사진으로 같은 절차(가짜 제작 → 학습 → 봉인 검증)를 다시 돌려 확인해야 한다", K.blue);
  s.addNotes("[말할 것] 현장 사진에는 AI Hub 원본이 없으므로 지문 DB를 끄고 다시 재 봤습니다. 균형 정확도가 1.4~2.3%p 내려가지만 핵심 성능은 유지됩니다. 형식 게이트까지 끈 결과도 재고 있습니다. 약점도 분명합니다. 선명화하거나 WebP로 저장한 실제 사진을 가짜로 많이 판정합니다. 그래서 현장에서는 그 농장 사진으로 같은 절차를 다시 돌려 확인해야 합니다.\n\n[노트]\n- 지문 DB 끄기 실험(10-20, 계획 사전 고정, 서술용, logs/ablation_원본대조_2026-10-20/): 나단11 코드 수정 없이 원본 대조 입력만 뺌. R2 .964→.945, R3 .962→.948, R4 .947→.924, R5 .944→.921. 실제 오탐 R5 1.4%→5.1%. 가짜 탐지 기여는 작음(부분 생성 82.3→79.2%).\n- 처리 변형 실제(R3 실제 429장 다시 저장): sharp(선명화 후 Q92) 90.7%, webp 37.5%, c444 5.4%, png 2.1%, area 2.1% 오탐(제출 모델). sharp·area는 형식 게이트 안.\n- 형식 게이트 끄기(ABL2, 11-02 계획 고정 커밋 6333b739): 판 E = 게이트 끔(배포 형식 전용 전문가 미적용), 판 F = 지문 DB + 게이트 모두 끔. 결과는 R3~R5 확보 후 반영.\n- 처음 보는 온실: AI Hub 596의 44농장이 모두 학습에 쓰여 새 농장 검증 자료가 없음(카탈로그 7.070) → 미실시.");
}

// ================= 12. AI와 팀 =================
{
  const s = body("함께 일한 방식", "판단과 설계는 팀이, 실행과 초안은 AI가 맡았다", K.dark);
  const col = (x, head, sub, items, fill, hc) => {
    RR(s, x, 1.9, 5.95, 3.95, fill);
    T(s, head, { x: x + 0.35, y: 2.05, w: 5.3, h: 0.45, fontSize: 20, bold: true, color: hc });
    T(s, sub, { x: x + 0.35, y: 2.5, w: 5.3, h: 0.35, fontSize: 13, color: K.sub });
    T(s, bl(items, 16, K.ink, 8), { x: x + 0.35, y: 3.0, w: 5.3, h: 2.8 });
  };
  col(0.6, "팀이 판단·설계", "결정과 책임", ["문제 정의와 채택 기준(실행 전 고정)", "규정 경계 판단", "현장형 정리(W2 제거 등)", "제출 후보·최종본 선택", "AI Hub 데이터 선택·이용 신청"], K.greenT, K.green);
  col(6.78, "AI가 수행", "Claude · Codex(코딩 도우미), ChatGPT(이미지 편집)", ["코드 작성과 실험 실행", "가설·실험 제안", "GPT 편집 가짜 이미지 생성", "문서 초안"], K.tint, K.sub);
  take(s, "예측·판정 단계에는 AI 도우미를 쓰지 않는다 — 제출 코드만으로 답을 만든다", K.green);
  s.addNotes("[말할 것] AI 도구를 많이 썼습니다. 다만 무엇을 할지, 무엇을 채택할지, 규정 경계를 어디에 둘지는 팀이 정했고, AI는 코드와 실험 실행, 제안, 초안을 맡았습니다. 예측 단계에는 AI 도우미를 쓰지 않습니다.\n\n[노트]\n- 팀 확인(11-02): 팀 = 문제 정의, 채택 기준, 규정 경계, W2 제거·현장형 정리, 제출 후보·최종본 선택, AI Hub 선택·신청 / AI = 코드, 실험 실행, 가설·실험 제안, 문서 초안(+ ChatGPT 이미지 생성).\n- 개발 도구: Anthropic Claude, OpenAI Codex(정형 README ①). 이미지 가짜 제작에 ChatGPT 이미지 편집(이미지 README_data 3절).");
}

// ================= 13. 정리 =================
{
  const s = pres.addSlide({ masterName: "DARK" });
  T(s, "정리", { x: 0.9, y: 0.7, w: 11, h: 0.6, fontSize: 30, bold: true, color: K.white });
  const col = (x, head, items, c) => {
    T(s, head, { x, y: 1.6, w: 5.6, h: 0.45, fontSize: 18, bold: true, color: c });
    T(s, bl(items, 16, K.white, 10), { x, y: 2.15, w: 5.6, h: 4.0 });
  };
  col(0.9, "한 것", [
    "정형: '어제'가 하루 수준 오차의 답임을 찾고, 대회 표식을 뺀 현장형 구조로 최종 답안",
    "정형: 어제 정보를 끈 성능을 재서 기여를 밝힘",
    "이미지: 가짜를 직접 만들어 학습·검증하는 절차",
    "이미지: 대회용 장치를 끈 성능을 재서 의존 정도를 밝힘",
  ], "A9C9B9");
  col(7.0, "향후 계획 (아직 하지 않음)", [
    "정형: 달력 날짜 기반 계절 지표로 교체, 실제 달력 어제로 재검증",
    "이미지: 처음 보는 온실 사진으로 검증",
    "이미지: 휴대폰 사진 처리(선명화·WebP)에 강하게 재학습",
  ], "F2C14E");
  s.addNotes("[말할 것] 정리하겠습니다. 한 것과 앞으로 할 것을 나눠 말씀드립니다. 왼쪽은 실제로 한 것이고, 오른쪽은 아직 하지 않은 계획입니다. 감사합니다.\n\n[노트]\n- 재현: 정형은 저장 중간값으로 24회차 업로드본과 바이트 단위 일치(check ALL PASS), 전체 재학습 시 EC 최대 .013 차이(TabPFN GPU→CPU). 이미지는 제출 ZIP CPU 재추론 공개 21/21, 순서 독립(정형·이미지 README ⑥).\n- 향후 계획 항목은 모두 미실시입니다.");
}

// ================= 부록 =================
{
  const s = pres.addSlide({ masterName: "DARK" });
  T(s, "부록", { x: 1.1, y: 2.6, w: 11, h: 0.9, fontSize: 40, bold: true, color: K.white });
  T(s, "자기 점검 · 운영위원회 공통 항목 1~3 · 팀별 항목 (사전심사용, 발표하지 않음)", { x: 1.1, y: 3.55, w: 11, h: 0.5, fontSize: 16, color: "C9DCD2" });
  T(s, bl(["A1 자기 점검 — 정형", "A2 자기 점검 — 이미지", "A3 공통 1: 현장에서 바뀌는 처리", "A4 공통 2: 성능 수치의 출처 + 정정", "A5 공통 3 ①②: 외부 서비스·환경", "A6 공통 3 ③④ · 팀별 이미지 3: 이용조건·지문 DB", "A7 팀별 정형 1: TabPFN 0.3·0.4", "A8 팀별 이미지 2: 장치가 작동하는 입력"], 13, "C9DCD2", 4), { x: 1.1, y: 4.3, w: 11, h: 2.9 });
  s.addNotes("부록은 발표 시간에 설명하지 않고 사전심사에서 확인하는 자료입니다. 각 장의 상세는 발표자 노트에 있습니다. '해당 없음', '미실시', '미확인'은 운영위원회 안내에 따른 표기입니다.");
}
function appx(code, title) {
  const s = pres.addSlide({ masterName: "BODY" });
  R(s, 0.6, 0.42, 0.12, 0.3, K.gray);
  T(s, code, { x: 0.82, y: 0.42, w: 8, h: 0.3, fontSize: 13, bold: true, color: K.gray, valign: "middle" });
  T(s, title, { x: 0.6, y: 0.78, w: 12.13, h: 0.7, fontSize: 24, bold: true, color: K.ink, valign: "middle" });
  return s;
}
const AT = { x: 0.6, y: 1.8, w: 12.13, fontSize: 12 };

{
  const s = appx("A1 · 정형", "자기 점검 1~6 — 정형");
  table(s, [
    ["문항", "답 요지", "위치"],
    ["1. 제출 코드·모델과 같은가", "같음. 저장 중간값으로 24회차 업로드본과 바이트 단위 일치, check ALL PASS. 전체 재학습 시 EC 최대 .013 차이(TabPFN GPU→CPU)", "정형 README ⑥"],
    ["2. 예측 시점 정보만", "예. 평가 행은 같은 온실의 그 시각까지·앞 일차 기록만. 절단 3곳·격리 2개 검사 ALL PASS", "README ②③"],
    ["3. 대회 자료 규칙성 의존", "계절 지표의 2차 구간 날짜 복원(외기 쌍둥이), 어제 찾기 링커(현장=달력). 어제 이어 붙이기 끈 성능: 온도 +3~10%, EC +8~31%(ABL1). 계절 지표 뺀 성능: 미실시", "A3, 본문 7"],
    ["4. 공개 리더보드 사용처", "제출물 비교·선택(여러 부품을 제출 비교로 채택). 상수 결정: TabPFN 비중 0.4 한 건", "README ⑤, A7"],
    ["5. 외부 자료·모델·AI 도구", "외부 데이터 없음. TabPFN v2 가중치(Prior Labs License = Apache 2.0 + 출처 표시, 11-02 원문 확인). 코딩 도우미 Claude·Codex", "README ①, A6"],
    ["6. AI / 팀", "팀: 문제 정의·채택 기준·규정 경계·현장형 정리·제출 선택 / AI: 코드·실험 실행·제안·초안", "본문 12"],
  ], Object.assign({}, AT, { colW: [2.6, 7.6, 1.93], rowH: 0.66 }));
  s.addNotes("1. 재현: 정형 README ⑥. produce 결과 = 제출 파일 바이트 일치, ZIP 새로 풀어 produce도 일치. 전체 재학습: EC 구성원·계절표·온도 NX는 저장본과 같고, HR5 학습 입력만 TabPFN GPU→CPU 차이(최대 .0096) → 최종 답안 온도 최대 .0016·EC 최대 .013.\n2. 평가 행 특징은 같은 온실의 현재·이전 입력만, 학습 행은 MASK. 중간값 단계 절단·격리 검사 ALL PASS(README ③).\n3. 상세는 A3. 링커는 대회 자료에서 '어제'를 찾는 장치로, 현장에서는 달력으로 정해지므로 필요 없음. 계절 지표: EC/season.py, README ② EC.\n4. 상세는 README ⑤ 표의 '제출 비교' 행들. TabPFN 0.4는 A7.\n5. TabPFN 가중치 라이선스: Hugging Face Prior-Labs/TabPFN-v2-reg LICENSE.txt(11-02 확인) — 배포·제공 시 'Built with PriorLabs-TabPFN' 표시와 라이선스 사본 동봉 조건. 최종 검증 ZIP에는 이 표시가 없음(README ①은 '원문 대조는 하지 않음'으로 기재).\n6. 본문 12쪽.");
}
{
  const s = appx("A2 · 이미지", "자기 점검 1~6 — 이미지");
  table(s, [
    ["문항", "답 요지", "위치"],
    ["1. 제출 코드·모델과 같은가", "같음. 제출 ZIP을 CPU로 다시 풀어 공개 21장 21/21, 정순·역순 결과 차 0", "이미지 README ⑥"],
    ["2. 예측 시점 정보만", "예. 사진 한 장만 입력, 사진끼리 정보 교환 없음, 비공개 평가 자료 미사용", "README ③"],
    ["3. 대회 자료 규칙성 의존", "지문 DB(AI Hub 출처), 형식 게이트(JPEG Q92·4:2:0). 지문 DB 끈 성능 R3~R5 −1.4~−2.3%p. 형식 게이트 끈 성능: 판정 진행 중. 처음 보는 온실: 미실시", "A3, A8, 본문 11"],
    ["4. 공개 리더보드 사용처", "제출물 비교·선택(최종안 나단11, 회차별 후보 채택). 상수·문턱 결정에는 쓰지 않음", "README ⑤"],
    ["5. 외부 자료·모델·AI 도구", "AI Hub 596, CLIP(MIT), SD1.5·SDXL(OpenRAIL), Qwen(Apache 2.0), ChatGPT 편집", "A5, A6"],
    ["6. AI / 팀", "본문 12와 같음", "본문 12"],
  ], Object.assign({}, AT, { colW: [2.6, 7.6, 1.93], rowH: 0.66 }));
  s.addNotes("1. 팀원 노트북 CPU(fp32, 10-17): 보관 중인 제출 ZIP을 그대로 풀어 공개 21장 21/21, 정순·역순 결과 차 0.0(README ⑥, logs/추론재현_공개21_집CPU_2026-10-17/). 서버 공식 self_check PASS.\n2. README ③: 이미지마다 독립 계산, 8장 묶음 GPU 계산으로 인한 수치 차 최대 .008. 시험 입력 통계를 모으는 분석은 규정 취지상 하지 않음.\n3. 지문 DB 끄기(10-20): R3 .962→.948, R4 .947→.924, R5 .944→.921. 형식 게이트 끄기(ABL2, 계획 커밋 6333b739) 결과 반영 예정. 새 농장 검증 불가(카탈로그 7.070).\n4. 리더보드: 최종안 선택(나단11), 회차별 후보 채택. E_gpt rc 제외는 내부 자료로만 결정(README ⑤).\n5. 상세는 A5·A6.");
}
{
  const s = appx("A3 · 공통 1", "자기 점검 2·3번 — 현장에서 바뀌는 처리");
  table(s, [
    ["미션", "대회 자료에 기대는 처리", "현장에서는", "뺀 성능"],
    ["정형", "어제 찾기 링커(기록이 섞여 있어서 필요)", "달력으로 어제가 정해짐 → 링커 불필요", "어제 이어 붙이기 끔: 온도 +3~10%, EC +8~31%"],
    ["정형", "계절 지표의 2차 구간 날짜 복원(외기 쌍둥이)", "달력 날짜를 그대로 씀", "미실시"],
    ["이미지", "지문 DB 원본 대조(AI Hub에 원본이 있음)", "원본이 없어 거의 작동 안 함 → 기본 판별+전문가 결과 그대로", "R3~R5 −1.4~−2.3%p"],
    ["이미지", "형식 게이트(JPEG Q92·4:2:0)", "형식이 다르면 전문가 4종이 꺼짐", "판정 진행 중"],
    ["이미지", "학습 가짜를 배포 형식으로 맞춘 처리", "그 현장의 저장 형식으로 맞춰 다시 학습", "미실시"],
  ], Object.assign({}, AT, { colW: [1.1, 4.0, 4.1, 2.93], rowH: 0.7 }));
  T(s, "학습 자료와 출처가 다른 자료로 확인한 결과: 정형 — 예측 사슬 식을 다른 49개 온실에서 외부 확인(48곳 개선). 이미지 — 미실시(모든 자료가 AI Hub 596)", { x: 0.6, y: 6.25, w: 12.13, h: 0.5, fontSize: 12, color: K.sub });
  s.addNotes("운영위 공통 1: 최종 답안에서 실제로 쓴 구성 요소 기준. 정형 상세: README ②·④, ABL1(카탈로그 7.069). 이미지 상세: 연구실 답변 초안 [이미지 2], 지문 DB 끄기 실험(logs/ablation_원본대조_2026-10-20/), ABL2 계획(집/클로드/research/abl2_PLAN.md).\n정형 다른 온실 외부 확인: 예측 사슬 식을 학습 자료의 다른 49개 온실에서 확인해 48곳 개선(정형 README ⑤).");
}
{
  const s = appx("A4 · 공통 2", "성능 수치의 출처 — 이미지 (+ 정정)");
  table(s, [
    ["수치", "낸 자료", "선택에도 썼나", "학습 자료와 겹침"],
    ["Accuracy .97 · 환산 25.59", "운영진 비공개 시험", "예: 최종안·회차별 후보 선택", "미확인(접근 불가)"],
    ["균형 정확도 R3 .962 · R4 .947 · R5 .944", "봉인 묶음(사진 자체는 학습 미사용)", "예: 후보 채택 판정에 각 1회", "농장·촬영 묶음 모두 겹침, 원본 겹침 0"],
    ["균형 정확도 R2 .964", "봉인 묶음", "예", "같은 원본 겹침: 실제 1,090장의 원본으로 만든 가짜가 학습에"],
    ["공개 21장 21/21", "대회 공개 샘플", "예: 초기 후보 선택 지표(5회 이상)", "학습에 없음"],
    ["끄기 실험 수치", "위 봉인 묶음", "아니오(서술용)", "위와 같음"],
  ], Object.assign({}, AT, { colW: [3.2, 3.0, 3.0, 2.93], rowH: 0.6 }));
  RR(s, 0.6, 5.55, 12.13, 1.2, K.amberT);
  T(s, "[정정] 최종 검증 자료 data/README_data.md 1절의 \"봉인 검증용은 촬영 묶음이 겹치지 않게 했습니다\"는 사실과 다릅니다. 봉인 묶음의 사진은 학습에 쓰지 않았지만 농장·촬영 묶음은 모두 학습과 겹칩니다. 원본 겹침이 없는 R3~R5를 대표값으로 씁니다.", { x: 0.85, y: 5.62, w: 11.7, h: 1.05, fontSize: 13, color: K.ink, valign: "middle" });
  s.addNotes("근거: 연구실 클로드 매니페스트 대조(11-02, 연구실/클로드/code/im75_sealed_overlap_check.py), 집 클로드 농장 집계(카탈로그 7.070).\n- 농장: R2 33곳, R3 11곳, R4·R5 각 6곳 — 모두 학습 농장(AI Hub 596 전체 44농장이 모두 학습에 쓰임).\n- 촬영 묶음: R2 70·R3 21·R4 7·R5 15개 묶음 모두 학습 자료에도 있음.\n- 같은 원본: R2만. R2 실제 1,862장 중 1,090장의 원본으로 만든 SDXL 가짜(5판)가 기본 판별기 학습에, 일부 원본의 GPT 가짜(gpt1 75·gpt2 41·gpt3 38)가 전문가 학습에 있음. R3~R5는 원본 겹침 0.\n- 정형 수치 출처표는 다음 판에 추가(검증기: 학습 기록을 가려 만든 EXT10·EXT12·DIAG10·EL1·P2LOO, 후보 채택 판정에 사용, 판정 기록 자체는 학습에서 가림).");
}
{
  const s = appx("A5 · 공통 3 ①②", "이미지 자료의 외부 서비스·외부 환경 처리");
  table(s, [
    ["환경", "단계", "국가"],
    ["ChatGPT(OpenAI) 이미지 편집", "AI Hub 사진 525장 + 편집 지시문 입력 → 가짜 600장. 최종 학습: 전문가 E_pg·E_ps(250), E_gr(72), E_gpt(600)", "미확인"],
    ["Google Drive", "AI Hub 원천 tar 306개 보관, 파생 이미지·학습 세트·제출 ZIP 보관", "미확인"],
    ["Google Colab", "정밀 지문 DB 계산(Drive 연결, tar 직접 처리), 추론 시간 측정", "미확인"],
    ["연구실 GPU 서버", "가짜 생성(SD·SDXL·Qwen 직접 실행)·특징·학습, 파생 이미지 보관", "국내"],
    ["연구실 PC", "12×16 지문 DB 계산, tar 추출·프로그램 가짜·저수준 특징", "국내"],
  ], Object.assign({}, AT, { colW: [2.9, 8.0, 1.23], rowH: 0.72 }));
  s.addNotes("근거: 연구실 답변 초안 [공통 3] ①②(연구실/클로드/문서/발표부록_운영위추가항목_답변초안_2026-11-02.md), 이미지 README_data 3절, data/gpt_generation/.\n- ChatGPT 입력 AI Hub 사진(중복 제외) 525장: gpt1 75(나머지 25장은 사진 없이 지시문만), gpt2 150, gpt3 150(결과 200), gptr 150. 묶음 사이 중복 없음.\n- SD1.5·SDXL·Qwen-Image-Edit은 외부 서비스가 아니라 연구실 GPU 서버에서 직접 실행(가중치만 Hugging Face에서 받음).\n- 처리·보관 국가를 확인하지 못한 곳은 '미확인'.");
}
{
  const s = appx("A6 · 공통 3 ③④ · 팀별 이미지 3", "이용조건 확인과 지문 DB 내용");
  T(s, "③ 이용조건 — 2026-11-02 원문 확인", { x: 0.6, y: 1.75, w: 6, h: 0.4, fontSize: 16, bold: true, color: K.blue });
  T(s, bl([
    "AI Hub 이용정책: 학습용 한정, 승인 없는 제3자 제공 금지, 국외 반출·국외 주체 이용은 별도 합의, 출처 표시",
    "우리 처리: 원본·파생 이미지는 제출물에서 빼고 해시로 대체(10-20). ChatGPT 입력·Drive/Colab 보관에 대한 별도 합의는 하지 않음",
    "모델: CLIP MIT, SD1.5 OpenRAIL-M, SDXL OpenRAIL++-M, Qwen Apache 2.0, TabPFN Prior Labs License",
    "출처를 확인하지 못한 자료: 해당 없음",
  ], 13, K.ink, 6), { x: 0.6, y: 2.2, w: 6.0, h: 4.5 });
  T(s, "④ 지문 DB 두 개 (팀별 이미지 3)", { x: 6.95, y: 1.75, w: 5.8, h: 0.4, fontSize: 16, bold: true, color: K.blue });
  table(s, [
    ["", "fp_db", "finefp_db"],
    ["사진별 값", "12×16 RGB 평균 + 세로 여부", "흑백 24×32 평균 + 라플라시안 에너지 192값"],
    ["장수", "152,043", "152,043(같은 사진)"],
    ["크기(xz)", "74.4MB", "119.0MB"],
    ["원천", "AI Hub 596 원천 tar 306개 전체", "같음"],
  ], { x: 6.95, y: 2.2, w: 5.78, colW: [1.2, 2.2, 2.38], fontSize: 12, rowH: 0.55 });
  T(s, bl(["사진·파일명은 넣지 않음(숫자 배열만). 사진별 아주 작은 축소본 성격", "포함 판단: 출처를 밝히고 포함(팀 결정 10-12). AI Hub 문의는 미실시"], 12, K.ink, 4), { x: 6.95, y: 5.15, w: 5.78, h: 1.4 });
  s.addNotes("AI Hub 이용정책 원문(https://www.aihub.or.kr/intrcn/guid/usagepolicy.do, 'AI 허브 개방 데이터' 항목, 11-02 확인):\n- '인공지능 학습모델의 학습용으로만 사용할 수 있습니다.'\n- 승인받지 않은 다른 법인·단체·개인에게 열람하게 하거나 제공·양도·대여·판매 금지.\n- '국외 반출을 위해서는 … 별도로 합의가 필요합니다.' 국외 소재 주체의 이용도 별도 합의.\n- 한국지능정보사회진흥원 사업 결과임을 밝힐 것(2차적 저작물 포함).\n- 특징값·축소본 등 파생 수치를 직접 다룬 조항은 없음. 데이터셋 596 페이지의 별도 조건은 미확인.\n우리 처리와 조항의 관계(해석은 하지 않음): ChatGPT 입력(525장), Drive 보관·Colab 처리는 별도 합의 없이 진행. 지문 DB는 10-12 팀 결정으로 출처를 밝히고 제출 모델에 포함. 원본·파생 이미지는 제출물에서 빼고 원본 대응 해시 표로 대체(10-20).\n지문 DB 배열(11-02 연구실 확인): fp_db = fp(152,043×16×12×3 uint8) + portrait. finefp_db = g(×768 uint8) + e(×192 uint8). 파일명 없음. 붉은 LED 사진도 포함. 만드는 코드: im11_fingerprint_db.py, colab_정밀지문.py.\n모델 라이선스(11-02 원문 확인): Hugging Face 모델 카드·GitHub LICENSE.");
}
{
  const s = appx("A7 · 팀별 정형 1", "TabPFN 비중 0.3·0.4 중 최종안을 고른 방법");
  table(s, [
    ["", "0.3안", "0.4안"],
    ["따뜻한 행(10℃ 이상) 비중", "기준 0.5 · CODEX 0.2 · TabPFN 0.3", "기준 0.4 · CODEX 0.2 · TabPFN 0.4"],
    ["8℃ 이하", "둘 다 기준 0.4 · CODEX 0.6 · TabPFN 0", "같음"],
    ["내부 검증(0.4 − 0.3)", "일반 날 DIAG10 −0.8%(0.4가 나음)", "추운 날 EXT10·EXT12 +0.1~0.2%(0.4가 나쁨)"],
    ["공개 리더보드(온도)", "7회차 .5193", "8회차 .5160 → 0.4 유지"],
  ], Object.assign({}, AT, { colW: [3.2, 4.47, 4.46], rowH: 0.62 }));
  T(s, bl([
    "내부 검증이 엇갈려 사전 채택 규칙(모든 검증기 같은 방향 + P < .025)을 통과하지 못함",
    "그래서 두 후보를 공개 리더보드 한 번의 비교로 정함 → 자기 점검 4번의 '상수 결정'에 해당. 한 회차 비교라 우연 가능성은 배제하지 못함",
  ], 13, K.ink, 6), { x: 0.6, y: 5.1, w: 12.13, h: 1.6 });
  s.addNotes("근거: 연구실 답변 초안 [정형 1], 8회차 패키지 온도/README.md 5절, 카탈로그 6.60~6.62·6.67·6.198·7.051, 제출/07회차·08회차 설명자료.\n- 두 후보 차이: 따뜻한 행에서 기준 몫 0.1을 TabPFN으로 옮긴 것만 다름. 실제 제출 답안 비교: 1,440행 중 1,225행 변경, RMS .036℃, 최대 .197℃.\n- 제출 전 내부 검증(새 표본 9~12, W30G 대비 W40G): DIAG10 −0.77/−0.78%(p_worse .089), EXT10 +0.14/+0.19%, EXT12 +0.17/+0.22%.\n- 사후 확인(리더보드를 본 뒤, TW1): 0.3 칸이 0.4 칸보다 DIAG10 2차 구간 +0.93% 나쁨(DIAG10만, 추운 행 비율 차이 있음).\n- 이후 데이터만으로 비중을 고르는 사전 고정 시험(B36)은 엇갈려 불통과 → 0.4 유지.\n- 내부 검증 스크립트 temp_user_w40g_check.py와 로그는 저장소·드라이브에서 찾지 못함(8회차 README 수치만 있음).");
}
{
  const s = appx("A8 · 팀별 이미지 2", "지문 DB와 형식 게이트가 작동하는 입력, 작동하지 않을 때");
  table(s, [
    ["장치", "작동하는 입력", "작동하지 않으면", "현장에서는"],
    ["원본 대조(지문 DB)", "AI Hub 152,043장 중 거의 같은 원본이 DB에 있는 사진", "규칙 미적용 → 기본 판별 + 전문가 확률 그대로", "원본이 없어 거의 항상 미작동"],
    ["형식 게이트", "JPEG + 4:2:0 + PIL 품질 92 양자화표가 완전히 같은 파일", "전문가 H·E_gr·E_gpt 미적용(기본 + E_pg·E_ps·E_L + 원본 대조만)", "휴대폰·카메라 원본은 대부분 게이트 밖"],
  ], Object.assign({}, AT, { colW: [2.2, 3.6, 3.4, 2.93], rowH: 0.95 }));
  T(s, bl([
    "지문 DB 끈 성능(10-20): R3 .962→.948, R4 .947→.924, R5 .944→.921 — 주로 실제 오탐 증가",
    "형식 게이트 끈 성능(ABL2): 판정 진행 중",
    "처리 변형 실제 오탐: 게이트 밖 png 2.1% · webp 37.5% · 4:4:4 5.4% / 게이트 안 area 2.1% · 선명화 90.7%",
    "다른 형식으로 저장한 가짜의 탐지율: 미실시",
  ], 13, K.ink, 6), { x: 0.6, y: 4.85, w: 12.13, h: 1.9 });
  s.addNotes("근거: 이미지 README ⑤(순서 2·4·5·7), 연구실 답변 초안 [이미지 2], logs/ablation_원본대조_2026-10-20/.\n- 원본 대조 규칙: R1 3.5<d1<20·gap>8·p≥.25 → p≥.9 / 정밀: dg1≤1.011·gap≥3.474일 때 국소 차이가 크면 가짜 확정(p≥.95), 작으면 진짜 확정(p≤.3).\n- 게이트 밖이면 이 세 전문가 없이 기본 판별기+E_pg·E_ps·E_L+원본 대조로 판정.\n- 선명화 사진은 형식이 배포 형식과 같아 세 전문가가 모두 적용되는데도 대부분 가짜로 판정됨 → 현장 휴대폰의 자동 선명화에 매우 약할 수 있음.\n- 끄기 실험 R2~R5의 실제 사진도 AI Hub 사진이라 원본이 DB에 있음 → 판 D가 현장(원본 없음)에 더 가까움. 다만 같은 AI Hub 농장·촬영 조건이라 다른 출처 사진 성능은 아님.");
}

// pptxgenjs는 슬라이드 마스터와 노트 마스터가 theme1.xml을 같이 쓰게 만든다.
// 한쇼는 이것을 '손상된 문서'로 보므로(11-02 최소 파일로 확인), 노트 마스터에 theme2.xml을 따로 붙인다.
async function splitNotesTheme(file) {
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const relP = "ppt/notesMasters/_rels/notesMaster1.xml.rels";
  const rel = await zip.file(relP).async("string");
  if (!rel.includes("theme1.xml")) throw new Error("노트 마스터 rels 형식이 예상과 다름");
  zip.file("ppt/theme/theme2.xml", await zip.file("ppt/theme/theme1.xml").async("string"));
  zip.file(relP, rel.replace("../theme/theme1.xml", "../theme/theme2.xml"));
  const ct = await zip.file("[Content_Types].xml").async("string");
  zip.file("[Content_Types].xml", ct.replace("</Types>", '<Override PartName="/ppt/theme/theme2.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/></Types>'));
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await splitNotesTheme(OUT);
  console.log("wrote", OUT);
})();
