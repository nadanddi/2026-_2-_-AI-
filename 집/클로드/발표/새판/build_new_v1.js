// 팜모니 AI 발표자료 새판 v1 (2026-11-02, 집 클로드)
// 실행: NODE_PATH=<pptxgenjs가 있는 node_modules> node build_new_v1.js <pptx 스킬 폴더>
// 내용 근거: ../발표자노트_초안_v1_2026-11-02.md, ../목차_v3_2026-11-02.md, ../ABL1_어제이어붙이기_끄기_결과_2026-11-02.md
const pptxgen = require("pptxgenjs");
const path = require("path");
const SKILL = process.argv[2];
const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js"));
const OUT = path.join(__dirname, "팜모니AI_발표자료_새판v1.pptx");

const FONT = "Malgun Gothic";
const HEX = { forest: "1E3B2F", leaf: "3F7D4E", berry: "D62839", tint: "EAF1EC", ink: "1A1A1A", gray: "5F6B66", line: "C9D6CD", white: "FFFFFF" };
const THEME = {
  name: "Farmoni", headFontFace: FONT, bodyFontFace: FONT,
  colors: { dk1: HEX.ink, lt1: HEX.white, dk2: HEX.forest, lt2: HEX.tint, accent1: HEX.leaf, accent2: HEX.berry, accent3: "7FA88A", accent4: "B7CDBD", accent5: HEX.gray, accent6: "E6A1A9", hlink: HEX.leaf, folHlink: HEX.gray },
};
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = "팜모니 AI 발표자료"; pres.author = "팜모니 AI";
const C = pres.SchemeColor;
const FOOT = "팜모니 AI · 제1회 전국 농과계 대학 농업 AI 경진대회";

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "tag", type: "body", x: 0.6, y: 0.28, w: 8, h: 0.35, fontSize: 12, bold: true, color: C.accent2, margin: 0, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.62, w: 12.1, h: 0.8, fontSize: 28, bold: true, color: C.text2, margin: 0, valign: "middle" }, text: "" } },
    { text: { text: FOOT, options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: 12.23, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});
pres.defineSlideMaster({
  title: "APPX", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "tag", type: "body", x: 0.6, y: 0.28, w: 8, h: 0.35, fontSize: 12, bold: true, color: C.accent1, margin: 0, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.62, w: 12.1, h: 0.8, fontSize: 24, bold: true, color: C.text2, margin: 0, valign: "middle" }, text: "" } },
    { text: { text: FOOT + " · 부록(사전심사용)", options: { x: 0.6, y: 7.0, w: 9, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: 12.23, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});
pres.defineSlideMaster({
  title: "DARK", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.3, fontSize: 44, bold: true, color: C.background1, margin: 0, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.0, w: 11.7, h: 1.2, fontSize: 18, color: C.accent4, margin: 0, valign: "top" }, text: "" } },
  ],
});

pres.defineSlideMaster({ title: "DARKPLAIN", background: { color: C.text2 }, objects: [] });
// ---------- helpers ----------
let N = 0; const nm = (s) => `${s}_${++N}`;
const T = (slide, text, o) => slide.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top", objectName: nm("txt") }, o));
const box = (slide, x, y, w, h, fill, o = {}) => slide.addShape(pres.shapes.ROUNDED_RECTANGLE, Object.assign({ x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: nm("card") }, o));
const circ = (slide, x, y, d, label, fill = C.text2, color = C.background1, fs = 16) => {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: nm("badge") });
  T(slide, label, { x, y, w: d, h: d, align: "center", valign: "middle", fontSize: fs, bold: true, color });
};
const arrow = (slide, x, y, w, h, fill = C.accent1) => slide.addShape(pres.shapes.RIGHT_ARROW, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: nm("arrow") });
const bl = (items, size = 14, color = C.text1) => items.map((t, i) => {
  const o = { bullet: { indent: 14 }, fontSize: size, color, paraSpaceAfter: 6 };
  if (i < items.length - 1) o.breakLine = true;
  if (Array.isArray(t)) return t.map((r, j) => ({ text: r.text, options: Object.assign({}, o, r.options, j < t.length - 1 ? { breakLine: false } : {}) }));
  return { text: t, options: o };
}).flat();
const rich = (runs) => runs; // [{text, options}]
const content = (title, tag, sec) => { const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sec }); s.addText(tag, { placeholder: "tag" }); s.addText(title, { placeholder: "title" }); return s; };
const appx = (title, tag) => { const s = pres.addSlide({ masterName: "APPX", sectionTitle: "부록" }); s.addText(tag, { placeholder: "tag" }); s.addText(title, { placeholder: "title" }); return s; };
const dark = (title, body, sec) => { const s = pres.addSlide({ masterName: "DARK", sectionTitle: sec }); s.addText(title, { placeholder: "title" }); if (body) s.addText(body, { placeholder: "body" }); return s; };
const AX = { catAxisLabelColor: HEX.gray, valAxisLabelColor: HEX.gray, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: HEX.ink, titleFontFace: "+mn-lt", titleFontSize: 12, titleColor: HEX.gray, valGridLine: { color: "E3E8E4", size: 0.75 }, catGridLine: { style: "none" } };
const tbl = (slide, rows, o) => {
  const hdr = rows[0].map((h) => ({ text: h, options: { bold: true, color: HEX.white, fill: { color: HEX.forest } } }));
  const body = rows.slice(1).map((r, i) => r.map((c) => (typeof c === "object" ? c : { text: String(c), options: { fill: { color: i % 2 ? HEX.white : HEX.tint } } })));
  slide.addTable([hdr, ...body], Object.assign({ fontFace: FONT, fontSize: 11, color: HEX.ink, border: { type: "solid", pt: 0.5, color: HEX.line }, valign: "middle", margin: 0.06, objectName: nm("table") }, o));
};

// =====================================================================
// 1. 표지
pres.addSection({ title: "시작" });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "시작" });
  T(s, "제1회 전국 농과계 대학 농업 AI 경진대회 · 현장 발표", { x: 0.8, y: 1.0, w: 11, h: 0.4, fontSize: 16, color: C.accent4 });
  s.addText("팜모니 AI", { placeholder: "title" });
  s.addText("정형: 딸기 배지 온도·EC 예측   |   이미지: 생성·편집 사진 판별", { placeholder: "body" });
  circ(s, 0.8, 5.35, 0.42, "", C.accent2);
  T(s, "발표 조나단 · 팀원 박찬유 · 반지원 · 김태경 · 김우진 · 지도교수 양명균", { x: 1.4, y: 5.35, w: 11, h: 0.42, valign: "middle", fontSize: 15, color: C.background1 });
  T(s, "2026. 11. 6 · 대구 EXCO", { x: 1.4, y: 5.85, w: 11, h: 0.35, fontSize: 13, color: C.accent4 });
  s.addNotes("팀 소개 한 줄.\n\n[노트] 발표 내용은 최종 검증에 제출한 답안 기준입니다. 정형 = submission_35e.csv(플랫폼 26회차, 24회차와 같은 답), 이미지 = 팜모니_나단11.zip(R21). 근거: 제출/최종검증/팜모니_정형/README.md ⑦, 팜모니_이미지/README.md 0·⑦.");
}

// 2. 두 미션 한눈에
{
  const s = content("두 미션 모두 '기본 모델 + 자료에서 찾은 정보원'으로 풀었다", "개요", "시작");
  const panel = (x, label, sub, nums) => {
    box(s, x, 1.65, 5.9, 3.35, C.background2);
    T(s, label, { x: x + 0.35, y: 1.85, w: 5.2, h: 0.45, fontSize: 20, bold: true, color: C.text2 });
    T(s, sub, { x: x + 0.35, y: 2.3, w: 5.2, h: 0.35, fontSize: 13, color: C.accent5 });
    nums.forEach((n, i) => {
      T(s, n[0], { x: x + 0.35 + i * 2.8, y: 2.85, w: 2.7, h: 1.1, fontSize: 48, bold: true, color: i === 0 ? C.accent2 : C.text2 });
      T(s, n[1], { x: x + 0.35 + i * 2.8, y: 3.95, w: 2.6, h: 0.8, fontSize: 13, color: C.text1 });
    });
  };
  panel(0.6, "정형 미션", "두 온실 배지 온도·EC, 시간별 예측", [["0.4608", "온도 RMSE (공개 점수)"], ["0.1148", "EC RMSE (공개 점수)"]]);
  panel(6.8, "이미지 미션", "딸기 시설재배 사진 진짜/가짜 판별", [["0.97", "Accuracy (공개 점수, R21)"], ["25.59", "환산점수"]]);
  const steps = ["기본 모델을 세운다", "오차를 뜯어보며 정보원을 찾는다", "실행 전에 정한 기준으로 붙일지 판정한다"];
  steps.forEach((t, i) => {
    const x = 0.6 + i * 4.1;
    circ(s, x, 5.45, 0.55, String(i + 1));
    T(s, t, { x: x + 0.7, y: 5.45, w: 3.2, h: 0.55, valign: "middle", fontSize: 15, bold: true, color: C.text2 });
  });
  T(s, "공개 점수는 플랫폼 리더보드 값 · 검증 수치의 출처는 부록 A5", { x: 0.6, y: 6.35, w: 12, h: 0.3, fontSize: 11, color: C.accent5 });
  s.addNotes("[말할 것] 두 미션 모두 기본 모델을 세운 뒤, 자료를 뜯어보며 정보원을 찾아 붙였습니다. 붙일지는 실행 전에 정한 기준으로 판정했습니다.\n\n[노트]\n- 정형 최종: 24회차 submission_35e.csv, 공개 점수 온도 .4608 · EC .1148. 26회차(10-16 18:22)에 같은 답을 다시 제출(마감 직전 운영진 1회 추가 허용). [정형 README ⑦]\n- 이미지 최종: 나단11, 플랫폼 18회차 업로드(10-16 15:30), R21 채점, SHA256 6fe04909…. [이미지 README 0·⑦]\n- 위 점수는 모두 공개 리더보드 값이며, 발표 중 검증 수치의 출처는 부록 A5에 정리했습니다.");
}

// ===================== 정형 =====================
pres.addSection({ title: "정형 미션" });
dark("정형 미션", "딸기 배지 온도·EC 예측 — 두 온실 · 평가 60기록 × 24시간", "정형 미션").addNotes("정형 미션 시작(전환용, 5초).");

// 3. 문제와 자료
{
  const s = content("하루 기록은 자정을 넘어 이어진다 — '어제'를 찾으면 정보가 는다", "정형 · 문제와 자료", "정형 미션");
  T(s, bl([
    "두 온실(F13·F47)의 시간별 배지 온도·EC 예측",
    "입력: 그 시각까지의 실내·외기·구동기 기록",
    "학습 400일(9,600행) · 평가 60기록, 모두 2차 구간",
    [{ text: "찾은 것 ① ", options: { bold: true, color: C.accent2 } }, { text: "하루 단위 기록이 섞여 있다" }],
    [{ text: "찾은 것 ② ", options: { bold: true, color: C.accent2 } }, { text: "같은 동이 두 온실 ID를 오간다(확실한 짝의 약 1/4)" }],
  ], 15), { x: 0.6, y: 1.7, w: 5.4, h: 4.6 });
  // 기록 이어짐 그림
  box(s, 6.4, 1.65, 6.3, 4.85, C.background2);
  T(s, "같은 동의 하루하루 (예시 그림)", { x: 6.7, y: 1.8, w: 5.5, h: 0.35, fontSize: 12, bold: true, color: C.accent5 });
  const days = [["F13", "1일"], ["F47", "2일"], ["F13", "3일"], ["F13", "4일"]];
  days.forEach((d, i) => {
    const x = 6.75 + i * 1.5;
    box(s, x, 2.5, 1.2, 1.3, d[0] === "F13" ? C.text2 : C.accent1);
    T(s, d[0], { x, y: 2.62, w: 1.2, h: 0.45, align: "center", fontSize: 16, bold: true, color: C.background1 });
    T(s, d[1], { x, y: 3.1, w: 1.2, h: 0.4, align: "center", fontSize: 13, color: C.background1 });
    if (i < 3) arrow(s, x + 1.22, 2.97, 0.26, 0.36, C.accent2);
  });
  T(s, "→ 전날 23시와 오늘 0시 센서값이 이어진다", { x: 6.75, y: 4.05, w: 5.8, h: 0.4, fontSize: 14, bold: true, color: C.text2 });
  T(s, bl(["대회 자료에서는 날짜가 섞이고 ID가 바뀌어 '어제'가 숨겨져 있다", "현장에서는 달력으로 어제가 정해진다"], 13), { x: 6.75, y: 4.6, w: 5.8, h: 1.6 });
  s.addNotes("[말할 것] 처음에는 시간별 회귀 문제로 봤습니다. 오차를 분석하다가 하루 기록이 서로 이어진다는 것을 알았습니다. 0시 센서값이 전날 23시와 이어지므로 '어제'를 찾을 수 있으면 정보가 늘어납니다.\n\n[노트]\n- 학습 자료: F13·F47 학습 기록 400일, 9,600행. [정형 README ④]\n- 기록 번호 = 문제설명서의 '온실별 상대 일차'. 179 미만 1차 구간, 이상 2차 구간. 평가 기록은 모두 2차 구간. [정형 README 용어]\n- 학습 정답으로 이어짐이 확실한 짝 가운데 약 4분의 1이 다른 온실 ID였습니다 → 같은 동이 ID를 오간다고 판단. [정형 README ③]\n- 이 구조는 대회 자료를 만드는 과정에서 생긴 성질이며 현장에서는 생기지 않습니다. 다룬 방법은 9쪽과 부록 A2.\n- 오른쪽 그림은 개념을 보여 주는 예시이며 실제 기록 번호가 아닙니다.");
}

// 4. 3단계 구조
{
  const s = content("기본 예측 → 어제 이어 붙이기 → 작은 보정의 3단계", "정형 · 전략", "정형 미션");
  const st = [
    ["1", "기본 예측", ["온도: 물리식·트리·TabPFN 섞기", "EC: 트리·신경망·TabPFN + 하루 수준 보정(HR5)"], "60 / 60기록"],
    ["2", "어제 이어 붙이기", ["링커가 '같은 동의 어제'를 찾음", "어제 정답 → 앵커 / 어제 예측 → 사슬"], "40 / 60기록"],
    ["3", "작은 보정", ["고EC 동 추운 시간 +0.25℃까지(TX3)", "학습 EC 범위로 자르기"], "13 / 60기록(TX3)"],
  ];
  st.forEach((v, i) => {
    const x = 0.6 + i * 4.2;
    box(s, x, 1.7, 3.75, 3.9, i === 1 ? C.text2 : C.background2);
    const fg = i === 1 ? C.background1 : C.text2;
    circ(s, x + 0.3, 1.95, 0.6, v[0], i === 1 ? C.accent2 : C.text2);
    T(s, v[1], { x: x + 1.05, y: 1.95, w: 2.6, h: 0.6, valign: "middle", fontSize: 20, bold: true, color: fg });
    T(s, bl(v[2], 14, i === 1 ? C.background1 : C.text1), { x: x + 0.3, y: 2.85, w: 3.2, h: 1.7 });
    T(s, v[3], { x: x + 0.3, y: 4.75, w: 3.2, h: 0.5, fontSize: 20, bold: true, color: i === 1 ? C.accent4 : C.accent2 });
    if (i < 2) arrow(s, x + 3.8, 3.4, 0.35, 0.5);
  });
  T(s, "기록 수 = 그 단계가 평가 예측을 바꾼 기록 수(제출 당시 중간 파일 비교). 어제 이어 붙이기 = 앵커 23 + 사슬 17", { x: 0.6, y: 5.85, w: 12, h: 0.35, fontSize: 11, color: C.accent5 });
  s.addNotes("[말할 것] 1단계가 성능의 대부분을 냅니다. 2단계는 어제를 찾을 수 있을 때만 작동합니다. 3단계는 자료에서 잰 작은 보정입니다.\n\n[노트]\n- 처리 순서: 온도 NX → 어제 앵커 → TX3 → 예측 사슬 / EC pre → HR5 → 어제 앵커 → 학습 EC 범위로 자르기. [정형 README ⑤]\n- 단계별 평가 예측 변화(제출 당시 중간 파일 비교): 어제 앵커 23기록 552행(변화 RMS .165℃), TX3 13기록 144행, 사슬 17기록 408행(RMS .109℃), EC HR5 60기록 전부, EC 앵커 23기록. [정형 README ⑤ 표]\n- 어제 이어 붙이기는 60기록 중 40기록에 작용했고, 나머지 20기록은 기본 예측과 TX3만으로 정해집니다.");
}

// 5. 기본 예측
{
  const s = content("추울수록 TabPFN을 줄이고, EC는 하루 수준을 따로 맞춘다", "정형 · ① 기본 예측", "정형 미션");
  box(s, 0.6, 1.65, 6.1, 4.95, C.background2);
  T(s, "온도 NX — 세 모델 섞기", { x: 0.9, y: 1.85, w: 5.5, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  T(s, bl(["BASE: 물리 선형식 + LightGBM 잔차", "CODEX: Ridge 물리 기준선 + LightGBM", "TabPFN v2: 사전학습 모델, 미세조정 없음", "실내 8℃ 이하에서 TabPFN 비중 0"], 14), { x: 0.9, y: 2.35, w: 2.9, h: 3.6 });
  const xs = ["4", "6", "8", "9", "10", "12", "14"]; const g = (t) => Math.min(1, Math.max(0, (t - 8) / 2));
  s.addChart(pres.charts.LINE, [{ name: "TabPFN 비중", labels: xs.map((v) => v + "℃"), values: xs.map((v) => +(0.4 * g(+v)).toFixed(2)) }], Object.assign({
    x: 3.85, y: 2.3, w: 2.75, h: 3.0, chartColors: [HEX.berry], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 6, showLegend: false,
    showTitle: true, title: "실내온도별 TabPFN 비중", valAxisMinVal: 0, valAxisMaxVal: 0.5, valAxisMajorUnit: 0.1, objectName: nm("chart_gate"),
  }, AX));
  T(s, "학습형 섞기는 추운 날 검증에서 +67~84% 악화 → 손으로 정한 게이트 유지", { x: 0.9, y: 5.55, w: 5.6, h: 0.75, fontSize: 12, color: C.accent5 });
  box(s, 7.0, 1.65, 5.7, 4.95, C.background2);
  T(s, "EC — 하루 수준이 통째로 어긋난다", { x: 7.3, y: 1.85, w: 5.2, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  T(s, "78~92%", { x: 7.3, y: 2.45, w: 5.2, h: 0.9, fontSize: 48, bold: true, color: C.accent2 });
  T(s, "EC 남은 오차 가운데 '하루 통째로 어긋난' 몫 (검증)", { x: 7.3, y: 3.35, w: 5.2, h: 0.4, fontSize: 13, color: C.text1 });
  T(s, bl(["구성원: ExtraTrees·LightGBM·MLP 0.8 + TabPFN 0.2", "HR5: 3시간마다 '그 기록의 하루 평균 잔차'를 TabPFN이 예측해 절반만큼 보정", "입력은 그 기록의 그 시각까지 값뿐"], 14), { x: 7.3, y: 3.95, w: 5.1, h: 2.4 });
  s.addNotes("[말할 것] 온도는 세 모델을 섞되 추운 시간에는 TabPFN 비중을 줄였습니다. 학습 범위 밖의 추운 행에서 TabPFN이 크게 틀렸기 때문입니다. EC는 하루 안에서 수준이 통째로 어긋나는 오차가 커서 하루 평균 수준을 따로 예측해 보정했습니다.\n\n[노트]\n- 섞는 식: g = clip((실내온도 − 8)/2, 0, 1), pred = 0.4·BASE + (0.2 + 0.4(1−g))·CODEX + 0.4·g·TabPFN. [정형 README ②] 그래프는 이 식의 TabPFN 비중(0.4·g)을 그린 것.\n- 학습 행 가중 0.2: 입력이 복원·보간된 흔적이 있는 행, 잡음 낀 날(학습 행만 판정). [정형 README ②]\n- 추운 행 근거: 학습형 섞기(STK1)가 추운 날 검증에서 크게 실패(≤ 8℃ 행 +67~+84%). 손으로 정한 '추우면 TabPFN 끄기'가 외삽 안전 장치였음을 확인. (학습 자료 검증기, 카탈로그 7.014)\n- EC 남은 오차의 65~77%가 고EC 기록 35개에 몰렸고, 하루 통째로 어긋난 몫이 78~92%. (학습 자료 검증기, 카탈로그 6.756) → HR5 설계 근거.\n- HR5: 기준 시각 0,3,…,21시마다 TabPFN v2가 그 기록의 하루 평균 잔차를 예측, NH = pre + 0.5·보정. 학습은 잠금 40일을 뺀 학습 기록 360개.\n- TabPFN v2: Prior Labs 사전학습 가중치 그대로, 학습 행 2,000개를 참고 예시로 넣어 예측.");
}

// 6. 어제 이어 붙이기
{
  const s = content("어제 23시 값은 오늘 새벽 예측의 가장 큰 단서다", "정형 · ② 어제 이어 붙이기", "정형 미션");
  // flow
  box(s, 0.6, 1.75, 2.3, 1.1, C.background2); T(s, "오늘 0시 입력", { x: 0.6, y: 1.75, w: 2.3, h: 1.1, align: "center", valign: "middle", fontSize: 16, bold: true, color: C.text2 });
  arrow(s, 3.0, 2.1, 0.45, 0.4);
  box(s, 3.55, 1.75, 2.6, 1.1, C.text2); T(s, "링커\n같은 동의 어제 고르기", { x: 3.55, y: 1.75, w: 2.6, h: 1.1, align: "center", valign: "middle", fontSize: 15, bold: true, color: C.background1 });
  box(s, 0.6, 3.25, 2.75, 1.55, C.background2);
  T(s, "어제에 실측 정답", { x: 0.8, y: 3.38, w: 2.4, h: 0.35, fontSize: 14, bold: true, color: C.accent2 });
  T(s, "앵커: 23시 정답으로 오늘 앞부분 보정", { x: 0.8, y: 3.78, w: 2.4, h: 0.9, fontSize: 14, color: C.text1 });
  box(s, 3.55, 3.25, 2.6, 1.55, C.background2);
  T(s, "어제가 평가 기록", { x: 3.75, y: 3.38, w: 2.3, h: 0.35, fontSize: 14, bold: true, color: C.accent2 });
  T(s, "사슬: 어제의 최종 예측을 이어 씀", { x: 3.75, y: 3.78, w: 2.3, h: 0.9, fontSize: 14, color: C.text1 });
  T(s, bl(["확신 p ≥ .8, 자정 실내 변화 ≤ 1.5℃일 때만", "평가 기록 사이 정보 전달은 사슬 하나뿐(주최 측 10/16 답변 2항)", "금지된 W2 규칙은 없음"], 13), { x: 0.6, y: 5.05, w: 5.6, h: 1.6 });
  const B = [0.7648, 0.6232, 0.5525, 0.4844, 0.4091, 0.3453, 0.3025, 0.2576, 0.1866, 0.1264, 0.157, 0.1661, 0.1655, 0.1566, 0.1999, 0.2032, 0.2233, 0.2263, 0.2145, 0.2008, 0.1897, 0.1721, 0.1991, 0.2355];
  s.addChart(pres.charts.LINE, [{ name: "β_h", labels: B.map((_, h) => String(h)), values: B }], Object.assign({
    x: 6.6, y: 1.7, w: 6.1, h: 4.4, chartColors: [HEX.forest], lineSize: 3, lineDataSymbol: "none", showLegend: false,
    showTitle: true, title: "어제 23시 값이 오늘 각 시각에 반영되는 비율 β_h (학습 정답으로 잼)", valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0", objectName: nm("chart_beta"),
  }, AX));
  T(s, "0시 .76 → 9시 .13, 이후 .16~.24에 머문다 · 가로축: 시각", { x: 6.6, y: 6.15, w: 6.1, h: 0.35, fontSize: 11, color: C.accent5 });
  s.addNotes("[말할 것] 배지 온도와 EC는 자정을 넘어 이어집니다. 어제 23시 값을 알면 오늘 새벽 예측이 크게 좋아집니다. 보정은 새벽에 크고 시간이 갈수록 약해집니다.\n\n[노트]\n- 앵커 식: pred_h = q_h + β_h·clip(A − 0.36 − q_0, −2, 2). A = 연결된 어제 학습 기록의 23시 배지온도 정답. β_h는 1차 구간 학습 정답으로 잰 시각별 비율(make_temp_v22_anchor.py 17행). [정형 README ②]\n- 적용 조건: 링커 확신 p ≥ .8 이고 실내 공기 자정 변화 |dT| ≤ 1.5℃. 또는 (TAG2) .5 ≤ p < .8 이면서 외기만 본 판단이나 실내만 본 판단이 같은 어제를 고른 경우.\n- 사슬 식: pred_h += 0.9·exp(−h/8)·β_h·clip(A − 0.36 − x_0, −2, 2), A = 앞 기록의 최종 23시 예측.\n- EC 앵커: pred_h = q_h + 0.8·p·(A − 그 시각까지 예측 평균).\n- 규정: 평가 기록 사이 정보 전달은 사슬 하나뿐. 주최 측 10/16 답변 2항(앞선 평가 기록에 대해 고정 규칙이 낸 값을 뒤 기록의 고정식 입력으로 쓰는 것)에 해당한다고 판단. [정형 README ③]\n- 링커 반사실 검사: 평가 기록을 후보에서 모두 빼도 정답 앵커 15기록의 어제는 그대로 → 앵커는 평가 기록의 존재와 무관. [정형 README ③]\n- W2(평가 자료에 앞 기록이 있는지로 갈리는 규칙)는 최종 답안에 없음. [정형 README ③]");
}

// 7. 검증
{
  const s = content("기준은 실행 전에 코드로 고정하고, 평가와 닮은 검증기로 판정했다", "정형 · 검증 방법", "정형 미션");
  tbl(s, [
    ["검증기", "가리는 방법", "왜"],
    ["DIAG10", "학습 기록을 10겹 무작위로", "전반적인 성능"],
    ["EL1 · P2LOO", "2차 구간 기록 묶음 · 한 기록씩", "평가 기록이 모두 2차 구간"],
    ["EXT10 · EXT12", "하루 최저 10·12℃ 미만인 날만", "평가에 추운 날이 섞임"],
  ], { x: 0.6, y: 1.7, w: 7.3, colW: [1.8, 3.0, 2.5], fontSize: 13, rowH: 0.55 });
  box(s, 0.6, 4.25, 7.3, 2.25, C.background2);
  T(s, "채택 기준", { x: 0.9, y: 4.4, w: 6.8, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  T(s, bl(["모든 검증기·시드에서 같은 방향으로 개선", "블록 부트스트랩 P(실제로는 악화) < .025", "판정은 같은 실행에서 재학습한 것끼리만 비교"], 14), { x: 0.9, y: 4.85, w: 6.8, h: 1.6 });
  T(s, "로컬 기준을 통과해 들어간 부품", { x: 8.3, y: 1.7, w: 4.4, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
  [["TX3", "P .0047"], ["예측 사슬", "P .0003"], ["사슬 문턱 p ≥ .5", "P .0029"]].forEach((v, i) => {
    const y = 2.25 + i * 0.95;
    box(s, 8.3, y, 4.4, 0.8, i === 1 ? C.text2 : C.background2);
    T(s, v[0], { x: 8.55, y, w: 2.5, h: 0.8, valign: "middle", fontSize: 15, bold: true, color: i === 1 ? C.background1 : C.text2 });
    T(s, v[1], { x: 10.9, y, w: 1.6, h: 0.8, valign: "middle", align: "right", fontSize: 18, bold: true, color: i === 1 ? C.accent4 : C.accent2 });
  });
  T(s, bl(["미래 입력 절단 3곳 · 다른 온실 격리 2개 규정 검사 ALL PASS", "기준을 다 못 채워 제출 비교로 채택한 부품도 있음(부록 A8)"], 13), { x: 8.3, y: 5.2, w: 4.4, h: 1.4 });
  s.addNotes("[말할 것] 평가 기록이 모두 2차 구간이고 추운 날이 섞여 있어서, 그 구조를 흉내 낸 검증기를 따로 만들었습니다. 기준을 실행 전에 정해 코드로 커밋한 뒤 실행했습니다.\n\n[노트]\n- 로컬 기준 통과 부품: TX3(44칸 모두 개선, P .0047), 사슬(P .0003), 사슬 문턱 p ≥ .5(P .0029). [정형 README ⑤ 표]\n- 기준을 다 채우지 못해 제출 비교로 채택한 부품: NX 섞기 비중, 계절 지표, 앵커, TAG2, CHT, HR5, EC 앵커. 그대로 밝힙니다(부록 A8).\n- 검증 세계 주의: 판정 기록의 입력이 학습 특징의 이력에 섞이면 추운 날 성적이 낙관적이 됨을 확인 → 판정은 같은 실행에서 재학습한 것끼리만 비교. (카탈로그 7.034)\n- 규정 검사: checks_v35e.py check — F13 206일 4시, F13 237일 3시, F47 202일 5시 뒤 입력을 크게 바꿔도 그 이전 예측이 비트 단위로 같음, 다른 온실 평가 입력을 바꿔도 예측이 같음. 중간값 단계도 같은 검사 ALL PASS. [정형 README ③]\n- 검증기는 같은 학습 자료 안에서 기록을 나눈 것이고 일부는 값 선택에도 썼습니다(부록 A5).");
}

// 8. 개선 과정
{
  const s = content("26번의 제출: 사전 기준을 통과한 변경만 점수로 옮겨졌다", "정형 · 개선 과정과 실패", "정형 미션");
  const tmp = [.7450, .6666, .5624, .5697, .5456, .5251, .5193, .5160, .5127, .5126, .5089, .5012, .4883, .4864, .4853, .4861, .4817, .4922, .4706, .4754, .4685, .5000, .4665, .4608, .4845, .4608];
  const ec = [.2442, .2287, .2055, .2055, .2134, .2055, .2774, .1455, .1384, .3538, .1305, .1271, .1279, .1077, .0979, .1225, .1170, .1173, .1153, .1153, .1153, .1153, .1153, .1148, .1148, .1148];
  const lab = tmp.map((_, i) => String(i + 1));
  s.addChart(pres.charts.LINE, [{ name: "온도 RMSE", labels: lab, values: tmp }], Object.assign({ x: 0.6, y: 1.6, w: 6.0, h: 3.0, chartColors: [HEX.forest], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 5, showLegend: false, showTitle: true, title: "온도 RMSE (공개 점수, 낮을수록 좋음)", valAxisMinVal: 0.4, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0", objectName: nm("chart_temp") }, AX));
  s.addChart(pres.charts.LINE, [{ name: "EC RMSE", labels: lab, values: ec }], Object.assign({ x: 0.6, y: 4.6, w: 6.0, h: 2.2, chartColors: [HEX.leaf], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 5, showLegend: false, showTitle: true, title: "EC RMSE (공개 점수) · 가로축: 제출 회차", valAxisMinVal: 0, valAxisMaxVal: 0.4, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0", objectName: nm("chart_ec") }, AX));
  const cards = [
    [C.text2, C.background1, "성공", "19회차 사슬(P .0003) .4706 · 21회차 사슬 문턱(P .0029) .4685 — 사전 기준 통과가 그대로 옮겨짐"],
    [C.background2, C.text1, "실패", "10회차 EC .3538(고EC 날 빼고 학습) · 22회차 온도 .5000(범위 밖 보정) — 기준 근소 미달·추운 영역 변경"],
    [C.background2, C.text1, "스스로 뺀 규칙", "W2를 주최 측 확인(10-16) 전 16회차에 제거 → EC .0979 → .1225를 감수"],
  ];
  cards.forEach((c, i) => {
    const y = 1.7 + i * 1.65;
    box(s, 7.0, y, 5.7, 1.45, c[0]);
    T(s, c[2], { x: 7.3, y: y + 0.15, w: 5.2, h: 0.4, fontSize: 16, bold: true, color: i === 0 ? C.accent4 : C.accent2 });
    T(s, c[3], { x: 7.3, y: y + 0.55, w: 5.2, h: 0.85, fontSize: 13, color: c[1] });
  });
  s.addNotes("[말할 것] 26번 제출하면서, 사전 기준을 엄격히 통과한 변경은 리더보드에서도 좋아졌고, 기준을 근소하게 못 넘었거나 추운 영역을 건드린 변경은 크게 어긋났습니다. 금지 확인을 받은 W2 규칙은 확인 전에 스스로 빼고 그 비용을 감수했습니다.\n\n[노트]\n- 그래프: 제출이력_정형_2026-10-20.md의 공개 점수. 26회차는 24회차와 같은 답(줄바꿈만 다름).\n- 실패: 10회차 EC .3538(고EC 날 빼고 학습), 22회차 .5000(범위 밖 보정 + 학습형 TX3), 20회차 .4754(BASE 이력 재연결), 25회차 .4845(실험 제출, 되돌림).\n- 성공: 19회차(사슬, 로컬 P .0003) .4706, 21회차(사슬 문턱, P .0029) .4685.\n- W2: 11~15회차에 포함. 현장형 정리 중 16회차(10-15)에 제거, EC .0979 → .1225. 주최 측이 10-16에 금지 확인, 이후 모든 제출에 없음. [제출이력]\n- 리더보드는 후보 비교에만 썼고 점수로 값을 계산하지 않았습니다. 예외: TabPFN 비중 0.4(두 후보 중 선택, 부록 A1).");
}

// 9. 현장 적용
{
  const s = content("어제 이어 붙이기를 끄면 오차가 온도 3~10%, EC 8~31% 늘어난다", "정형 · 농업적 해석과 현장 적용", "정형 미션");
  const labs = ["온도 EXT10", "온도 EXT12", "온도 DIAG10", "EC DIAG10", "EC EL1", "EC P2LOO"]; const vals = [10.4, 3.1, 3.4, 8.3, 30.5, 11.0];
  s.addChart(pres.charts.BAR, [{ name: "끄면 늘어나는 오차", labels: labs, values: vals }], Object.assign({
    x: 0.6, y: 1.6, w: 6.3, h: 4.5, barDir: "bar", chartColors: [HEX.berry], showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0"%"',
    showTitle: true, title: "끄면 늘어나는 RMSE (%, 학습 자료 검증기 · 2차 행)", valAxisMinVal: 0, valAxisMaxVal: 35, valAxisMajorUnit: 10, catAxisOrientation: "maxMin", objectName: nm("chart_abl"),
  }, AX));
  T(s, "ABL1 끄기 실험(11-02, 계획 사전 고정, 서술용) · 평가 자료 아님", { x: 0.6, y: 6.15, w: 6.3, h: 0.3, fontSize: 11, color: C.accent5 });
  const rows = [
    ["해석 ①", "배지 상태는 밤을 넘어 이어진다 — 전날 마지막 값이 새벽 예측의 가장 큰 단서"],
    ["해석 ②", "고EC 동은 추운 시간 배지 온도가 입력 기반 예측보다 높다(+0.25℃까지)"],
  ];
  rows.forEach((r, i) => {
    const y = 1.65 + i * 1.2;
    box(s, 7.3, y, 5.4, 1.05, C.background2);
    T(s, r[0], { x: 7.55, y: y + 0.12, w: 1.2, h: 0.4, fontSize: 14, bold: true, color: C.accent2 });
    T(s, r[1], { x: 8.65, y: y + 0.12, w: 3.9, h: 0.85, fontSize: 13, color: C.text1 });
  });
  box(s, 7.3, 4.15, 5.4, 2.35, C.text2);
  T(s, "현장에서는", { x: 7.55, y: 4.3, w: 5, h: 0.4, fontSize: 16, bold: true, color: C.accent4 });
  T(s, bl(["링커는 필요 없다 — 어제는 달력으로 정해진다", "사슬(어제의 내 예측 이어 쓰기)은 그대로 가능", "앵커는 전날 배지 센서 실측이 있을 때만"], 14, C.background1), { x: 7.55, y: 4.75, w: 5, h: 1.7 });
  s.addNotes("[말할 것] 현장에서는 어제가 달력으로 정해지므로 링커가 필요 없습니다. 어제의 내 예측을 이어 쓰는 사슬은 현장에서도 됩니다. 어제 실측 정답을 쓰는 앵커는 전날 배지 센서가 있어야 합니다. 이 차이를 숫자로 보였습니다.\n\n[노트]\n- ABL1(11-02 추가 실시, 서술용, 사전 고정 커밋 45ac69c5). 2차 행 RMSE, 일반 방법만 → 최종:\n  온도 EXT10 .8387 → .7599(+10.4%), EXT12 .9059 → .8791(+3.1%), DIAG10 .7388 → .7146(+3.4%)\n  EC DIAG10 .2522 → .2329(+8.3%), EL1 .2787 → .2135(+30.5%), P2LOO .1913 → .1723(+11.0%)\n  일반 방법만 = 온도 NX + TX3 / EC pre + HR5.\n- 온도 몫 분해: 사슬(현장에서도 가능) 0.4~3.1%p, 앵커 EXT10 8%p · DIAG10 3%p · EXT12 0(2차 판정 기록에 앵커 없음).\n- 해석 ①: β_h(학습 정답으로 잼)가 0시 .76 → 9시 .13, 이후 .16~.24. '밤사이 이어진다'는 이 데이터에서 본 결과이고, 배지 열·수분 특성에 대한 일반론은 팀의 해석입니다.\n- 해석 ②: TX3 0.25℃는 학습 정답으로 잰 고EC 동의 '배지온도 − 입력 기반 예측' 차. 로컬 기준 통과(P .0047). 원인(배지 조성·관수 등)은 미확인.\n- 대회 자료 성질에 기대는 부분: 링커, excl 특징, 계절 지표(외기 쌍둥이로 2차 날짜 복원), 사슬의 연결. 현장 대응은 부록 A2.\n- 미실시: 계절 지표까지 뺀 성능, 실제 어제로 거는 현장형 사슬 성능. 그래서 '현장에서 남는 성능 = 기본 + 사슬'은 검증 전 추론입니다.\n- 한계: 검증기는 같은 학습 자료 안에서 나눈 것, 2차 판정 기록 수가 적음(EXT10 9기록, EXT12 26, DIAG10 52). 상세: 집/클로드/발표/ABL1_어제이어붙이기_끄기_결과_2026-11-02.md");
}

// ===================== 이미지 =====================
pres.addSection({ title: "이미지 미션" });
dark("이미지 미션", "딸기 시설재배 사진 — 진짜(0)인가, 생성·편집된 가짜(1)인가", "이미지 미션").addNotes("이미지 미션 시작(전환용, 5초).");

// 10. 전략
{
  const s = content("기본 판별 위에 원본 대조와 유형별 전문가를 얹었다", "이미지 · 문제와 전략", "이미지 미션");
  const rows = [
    ["1", "기본 판별", "CLIP ViT-L/14 의미 특징 + 로지스틱 회귀 / 저수준 통계 38개(잡음·선명도·JPEG 격자) + 부스팅 → 두 확률의 평균", "모든 사진"],
    ["2", "원본 대조", "AI Hub 사진 152,043장의 지문 DB와 비교 — 원본이 있는데 국소적으로 다르면 가짜 쪽, 거의 같으면 진짜 쪽", "원본이 DB에 있을 때"],
    ["3", "유형별 전문가", "부분 생성 · 부분 스타일 · 국소 조각 · 필터 · GPT 재그리기 · GPT 편집 — 하나라도 문턱을 넘으면 가짜 쪽", "일부는 배포 형식일 때"],
  ];
  rows.forEach((r, i) => {
    const y = 1.7 + i * 1.6;
    box(s, 0.6, y, 12.1, 1.4, i === 1 ? C.text2 : C.background2);
    const fg = i === 1 ? C.background1 : C.text1;
    circ(s, 0.9, y + 0.4, 0.6, r[0], i === 1 ? C.accent2 : C.text2);
    T(s, r[1], { x: 1.75, y, w: 2.2, h: 1.4, valign: "middle", fontSize: 18, bold: true, color: i === 1 ? C.background1 : C.text2 });
    T(s, r[2], { x: 4.0, y: y + 0.2, w: 6.0, h: 1.0, valign: "middle", fontSize: 14, color: fg });
    T(s, r[3], { x: 10.2, y, w: 2.3, h: 1.4, valign: "middle", fontSize: 13, bold: true, color: i === 1 ? C.accent4 : C.accent2 });
  });
  T(s, "모든 계산은 이미지 한 장 단위로 독립 · 판정 임계값 0.5 · 문턱은 학습 자료에서 정함", { x: 0.6, y: 6.6, w: 12, h: 0.3, fontSize: 11, color: C.accent5 });
  s.addNotes("[말할 것] 사전학습 CLIP 특징과 잡음·JPEG 같은 저수준 통계를 반반 섞어 기본 판별을 만들었습니다. 기본 판별이 놓치는 유형은 작은 전문가들로 따로 잡았습니다.\n\n[노트]\n- 확률 계산 순서 1~7과 문턱은 이미지 README ⑤ 표(부록 A4·A5).\n- 전문가 문턱: 실제 사진을 0.25~0.4%만 뒤집는 값, 학습 자료(실제 풀)에서 정함.\n- 오른쪽 열 = 그 단계가 작동하는 조건. 원본 대조는 DB에 원본이 있다고 판단될 때만, 전문가 H·E_gr·E_gpt는 파일이 JPEG·4:2:0·Q92일 때만 켜집니다(형식 게이트).\n- 이미지끼리 정보를 주고받지 않으며 순서 독립 검사 OK. [이미지 README ③]");
}

// 11. 데이터
{
  const s = content("같은 원본으로 진짜와 가짜를 함께 만들어 '편집 흔적'을 배우게 했다", "이미지 · 데이터 확보·생성", "이미지 미션");
  [["17,160", "최종 학습 이미지(장)"], ["6", "문제설명서 가짜 유형 모두 직접 제작"], ["525", "ChatGPT 편집에 넣은 AI Hub 원본(장)"], ["152,043", "지문 DB 원천 사진(장)"]].forEach((v, i) => {
    const x = 0.6 + i * 3.1;
    box(s, x, 1.7, 2.85, 1.9, i === 0 ? C.text2 : C.background2);
    T(s, v[0], { x: x + 0.25, y: 1.85, w: 2.5, h: 0.95, fontSize: 40, bold: true, color: i === 0 ? C.background1 : C.accent2 });
    T(s, v[1], { x: x + 0.25, y: 2.8, w: 2.4, h: 0.7, fontSize: 13, color: i === 0 ? C.accent4 : C.text1 });
  });
  T(s, "진짜", { x: 0.6, y: 3.95, w: 5.8, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  T(s, bl(["AI Hub 「지능형 수직농장 통합 데이터(딸기)」(596), 이용 신청·승인", "공개 샘플과 촬영 환경이 가장 비슷한 공개 데이터셋", "촬영 묶음·품종·농장이 고르게 뽑음 → 배포 형식(768×1024, JPEG Q92, 4:2:0)으로 변환"], 14), { x: 0.6, y: 4.4, w: 5.9, h: 2.3 });
  T(s, "가짜", { x: 6.8, y: 3.95, w: 5.8, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  T(s, bl(["필터 · 합성 · 부분 생성 · 부분 스타일 · 전체 생성 · 전체 스타일", "SD1.5 · SDXL · Qwen-Image-Edit · 프로그램 처리", "ChatGPT 생성·편집 600장(25장은 문장만으로)", "처리 흔들기: 실제 사진을 다른 축소·JPEG로 재저장해 오탐 방지"], 14), { x: 6.8, y: 4.4, w: 5.9, h: 2.3 });
  s.addNotes("[말할 것] 공개 샘플이 21장뿐이라 학습 자료를 직접 만들었습니다. 공개 샘플과 촬영 환경이 가장 비슷한 공개 데이터셋을 찾아, 같은 사진으로 진짜와 가짜를 함께 만들었습니다. 그래서 모델이 사진 내용이 아니라 편집 흔적을 보게 했습니다.\n\n[노트]\n- 학습 이미지 17,160장(목록 17,164장 중 4장은 코드 규칙으로 제외). 기본 판별기는 실제 2,467 + 가짜 7,124. [이미지 README ①·④]\n- AI Hub 원천 tar 306개(Training 271, Validation 35)를 이용 승인 후 팀원 계정으로 받음. 대회 이미지에 맞춰 고르지 않고 데이터셋 안에서 고르게 뽑음. [이미지 data/README_data.md 1절]\n- ChatGPT에 넣은 AI Hub 원본: 서로 다른 525장(gpt1 75 · gpt2 150 · gpt3 150 · gptr 150, 겹침 0). 만든 이미지 600장, 그중 25장은 원본 없이 문장만으로. 11-02 기록 검산(부록 A6).\n- 재배포 제한 때문에 AI Hub 소재 16,197장은 파일 대신 원본 대응 해시로 냈고, 문장만으로 만든 967장은 파일로 냈습니다. [이미지 README ①]\n- 공개 샘플 21장은 학습에 넣지 않았고, 형식 실측·초기 구성 선택(5회 이상)·이후 21/21 유지 점검에 썼습니다.");
}

// 12. 검증·개선
{
  const s = content("실제 사진 오판이 늘지 않는 변경만 남겼다", "이미지 · 검증과 개선 과정", "이미지 미션");
  s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: ["팀원 기준", "나단1", "나단6", "나단9", "나단11(최종)"], values: [0.77, 0.90, 0.95, 0.97, 0.97] }], Object.assign({
    x: 0.6, y: 1.6, w: 6.4, h: 4.6, barDir: "col", chartColors: [HEX.forest, HEX.forest, HEX.forest, HEX.forest, HEX.berry], showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
    showTitle: true, title: "공개 리더보드 Accuracy", valAxisMinVal: 0.6, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0", objectName: nm("chart_img"),
  }, AX));
  T(s, "나단3은 추론 시간 초과로 반려(채점 없음)", { x: 0.6, y: 6.25, w: 6.4, h: 0.3, fontSize: 11, color: C.accent5 });
  [["봉인 검증 묶음 R2~R5", "촬영 묶음이 학습과 겹치지 않게 떼어 두고, 후보 판정 때만 열었다"], ["촬영 묶음 단위 교차검증", "같은 원본의 진짜·가짜가 다른 폴드로 갈리지 않게"], ["실패에서 고친 것", "GPT 편집 전문가 첫 판이 실제 오탐 기준을 근소하게 넘음 → 원인('진짜 확정' 사진)을 찾아 제외"]].forEach((v, i) => {
    const y = 1.7 + i * 1.6;
    box(s, 7.4, y, 5.3, 1.4, i === 0 ? C.text2 : C.background2);
    T(s, v[0], { x: 7.65, y: y + 0.15, w: 4.9, h: 0.4, fontSize: 15, bold: true, color: i === 0 ? C.accent4 : C.text2 });
    T(s, v[1], { x: 7.65, y: y + 0.58, w: 4.9, h: 0.75, fontSize: 13, color: i === 0 ? C.background1 : C.text1 });
  });
  s.addNotes("[말할 것] 같은 원본의 진짜와 가짜가 학습과 검증에 갈라지지 않게 촬영 묶음 단위로 나눴습니다. 후보는 봉인 묶음에서 '실제 사진 오판이 늘지 않는가'를 먼저 봤습니다.\n\n[노트]\n- 회차 흐름(성능_정리 6절): 팀원 기준 .77(고주파 CNN) → 나단1 .90(CLIP + 저수준 + 원본 대조) → 나단3 시간 초과 반려 → 나단6 .95(전문가 + 정밀 원본 대조) → 나단9 .97(+ GPT 재그리기 전문가) → 나단11 .97, 환산 25.59(+ GPT 편집 전문가).\n- 실패와 수정: E_gpt 첫 판은 GPT 가짜 26장 중 14장을 새로 잡았지만 실제 오탐 기준을 근소하게 넘어 불통과(합집합 +0.12%p, R5 −0.28%p). 새 오판 대부분이 '진짜 확정' 사진이었고 GPT 가짜 600장 중 진짜 확정은 0장 → 제외. 불통과를 본 뒤 같은 자료로 정한 변형이라 낙관적일 수 있음을 밝힙니다. [이미지 README ⑤]\n- 10-16 내부 불통과로 버린 후보: 필터 전문가 재학습(C1, R3 하락), 토큰 LR(A2, R3~R5 하락). [성능_정리 6절]\n- 최종 선택: 사전 규칙은 'LB ≥ .98이면 유지'. 두 판 모두 .97로 표시됐지만 더 세밀한 환산점수(23.82 → 25.59)와 내부 검증을 보고 나단11 유지. [이미지 README ⑤]\n- 봉인 R2~R5는 후보 채택 판정에 썼으므로 선택에 쓰인 자료이고, 학습과 같은 AI Hub 데이터셋입니다(부록 A5).");
}

// 13. 현장 한계
{
  const s = content("현장 사진엔 원본이 없다 — 지문 DB를 빼면 정확도 1.4~2.3%p 하락", "이미지 · 현장 적용 조건·한계", "이미지 미션");
  const lb = ["R2", "R3", "R4", "R5"];
  s.addChart(pres.charts.BAR, [{ name: "지문 DB 켬(최종)", labels: lb, values: [96.4, 96.2, 94.7, 94.4] }, { name: "지문 DB 끔", labels: lb, values: [94.5, 94.8, 92.4, 92.1] }], Object.assign({
    x: 0.6, y: 1.6, w: 6.6, h: 4.6, barDir: "col", barGrouping: "clustered", chartColors: [HEX.forest, HEX.berry], showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: HEX.gray,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", showTitle: true, title: "봉인 묶음 균형 정확도(%)", valAxisMinVal: 88, valAxisMaxVal: 98, valAxisMajorUnit: 2, objectName: nm("chart_fp"),
  }, AX));
  T(s, "지문 DB 끄기 실험(10-20, 계획 사전 고정, 서술용) · 공개 21장은 21/21 그대로", { x: 0.6, y: 6.25, w: 6.6, h: 0.3, fontSize: 11, color: C.accent5 });
  box(s, 7.6, 1.7, 5.1, 2.0, C.text2);
  T(s, "1.4% → 5.1%", { x: 7.85, y: 1.85, w: 4.7, h: 0.85, fontSize: 36, bold: true, color: C.background1 });
  T(s, "끄면 R5 실제 사진 오탐률 — 하락은 주로 오탐 증가에서", { x: 7.85, y: 2.75, w: 4.7, h: 0.8, fontSize: 13, color: C.accent4 });
  T(s, bl(["가짜 탐지 기여는 작다(부분 생성 82.3 → 79.2%)", "형식 게이트: JPEG·4:2:0·Q92가 아니면 전문가 3개가 꺼짐 — 끈 성능은 미실시", "다른 출처 사진으로 한 검증은 미실시"], 14), { x: 7.6, y: 4.0, w: 5.1, h: 2.6 });
  s.addNotes("[말할 것] 원본 대조는 AI Hub에 원본이 있는 사진에서만 작동합니다. 현장 사진에는 원본이 없으므로 이 부분이 빠진 성능으로 봐야 합니다. 빠지면 가짜 탐지는 거의 그대로지만 실제 사진 오탐이 늘어납니다.\n\n[노트]\n- 끄기 실험(10-20, 계획 사전 고정, 서술용): 균형 정확도 R2 .964 → .945, R3 .962 → .948, R4 .947 → .924, R5 .944 → .921. 공개 21장은 21/21 그대로. [이미지 README ⑤, logs/ablation_원본대조_2026-10-20/]\n- 처리 변형 실제(webp) 오탐 37.5% → 75.8%. 가짜 탐지 기여는 작음(부분 생성 82.3 → 79.2%, 부분 스타일 90.6 → 88.3%).\n- R1(12×16 색 지문)만 끄면 −0.4~+0.1%p로 작고, 차이는 대부분 정밀 대조의 '진짜 확정'에서 옵니다.\n- 형식 게이트: 파일이 JPEG·4:2:0·PIL Q92 양자화표일 때만 전문가 H·E_gr·E_gpt를 켭니다. 현장 사진 형식이 다르면 이 셋이 빠지며, 게이트를 끈 성능은 미실시(부록 A4).\n- 학습 자료와 출처가 다른 자료로 확인한 성능: 미실시(봉인 묶음도 같은 데이터셋).\n- 지문 DB는 사진별 아주 작은 축소본(12×16 컬러, 24×32 흑백)을 담고 있음. 이용조건 판단은 부록 A6.");
}

// ===================== 마무리 =====================
pres.addSection({ title: "마무리" });
// 14. AI vs 팀
{
  const s = content("AI는 속도를 냈고, 무엇을 쓸지는 팀이 실행 전에 정했다", "공통 · AI 도구와 팀의 판단", "마무리");
  const col = (x, head, items, darkc) => {
    box(s, x, 1.7, 5.9, 4.85, darkc ? C.text2 : C.background2);
    T(s, head, { x: x + 0.35, y: 1.9, w: 5.2, h: 0.5, fontSize: 20, bold: true, color: darkc ? C.background1 : C.text2 });
    T(s, bl(items, 15, darkc ? C.background1 : C.text1), { x: x + 0.35, y: 2.6, w: 5.2, h: 3.8 });
  };
  col(0.6, "팀이 판단·설계", ["문제 정의", "채택 기준(실행 전 고정, 모든 검증기 같은 방향)", "규정 경계: 예측 시점 이전 정보까지만", "W2 제거와 현장형 정리", "제출 후보와 최종본 선택", "AI Hub 데이터셋 선택·이용 신청"], true);
  col(6.8, "AI(Claude·Codex)가 수행", ["코드 작성", "실험 실행", "가설·실험 제안", "GPT 이미지 생성 작업", "문서 초안"], false);
  s.addNotes("[말할 것] AI 도구로 실험 속도를 높였습니다. 무엇을 채택하고 어디까지 쓸지는 팀이 실행 전에 기준으로 정했습니다.\n\n[노트]\n- 예측(추론)에는 AI 도우미를 쓰지 않습니다. [정형 README ①, 이미지 README ①]\n- 팀 판단의 예: '예측 시점 이전 정보까지만' 경계(10-13), W2를 주최 측 확인 전에 뺀 것(10-15), 실행 전 기준 고정 원칙. 날짜는 팀 작업 기록 기준.\n- 사용 도구: Anthropic Claude, OpenAI Codex, ChatGPT 내장 이미지 생성·편집(학습 자료 제작).");
}
// 15. 정리
{
  const s = pres.addSlide({ masterName: "DARKPLAIN", sectionTitle: "마무리" });
  T(s, "정리", { x: 0.8, y: 0.7, w: 11, h: 0.8, fontSize: 36, bold: true, color: C.background1 });
  const done = [["정형", "기본 예측 + 어제 이어 붙이기 — 규칙성 의존 몫을 숫자로 분리(온도 3~10%, EC 8~31%)"], ["이미지", "기본 판별 + 원본 대조 + 유형별 전문가 — 원본 대조 몫을 숫자로 분리(1.4~2.3%p)"], ["공통", "실행 전 기준 고정 · 규정 검사 · 재현 가능한 제출"]];
  done.forEach((d, i) => {
    const y = 1.8 + i * 1.0;
    circ(s, 0.8, y, 0.6, String(i + 1), C.accent2);
    T(s, d[0], { x: 1.6, y, w: 1.5, h: 0.6, valign: "middle", fontSize: 18, bold: true, color: C.background1 });
    T(s, d[1], { x: 3.1, y, w: 9.4, h: 0.6, valign: "middle", fontSize: 15, color: C.background1 });
  });
  T(s, "향후 과제 — 아직 하지 않은 것", { x: 0.8, y: 5.0, w: 11, h: 0.45, fontSize: 16, bold: true, color: C.accent4 });
  T(s, "현장형 사슬 검증  ·  계절 지표를 달력으로 바꾼 검증  ·  다른 출처 사진으로 이미지 검증  ·  형식 게이트를 끈 성능", { x: 0.8, y: 5.5, w: 11.7, h: 0.5, fontSize: 15, color: C.background1 });
  s.addNotes("[말할 것] 한 것과 향후 계획을 나눠 말합니다. 향후 항목은 모두 '아직 하지 않았다'고 분명히 말합니다.\n\n[노트]\n- 재현: 정형은 저장 중간값으로 24회차 업로드본과 바이트 단위로 재현, 전체 재학습 시 EC 최대 .013 차이(HR5 TabPFN을 GPU 대신 CPU로 돌린 차이). 이미지는 제출 ZIP을 CPU로 재추론해 공개 21/21. [정형 README ⑥, 이미지 README ⑥]\n- 향후 과제는 모두 미실시입니다.");
}

// ===================== 부록 =====================
pres.addSection({ title: "부록" });
dark("부록 — 자기 점검", "사전심사용 · 발표 시간에는 설명하지 않습니다 · 상세는 각 슬라이드 노트", "부록").addNotes("부록 표지. 운영위원회 11-01 팀별 안내에 따라 자기 점검·공통 항목을 슬라이드에 요지, 노트에 상세로 적었습니다. 최종 검증 자료에 이미 적은 내용은 파일·절 위치를 적었습니다.\n표기: [정형 README] = 제출/최종검증/팜모니_정형/README.md, [이미지 README] = 제출/최종검증/팜모니_이미지/README.md, [이미지 data] = 제출/최종검증/팜모니_이미지/data/README_data.md");

// A1
{
  const s = appx("자기 점검 1·4·5·6 — 정형", "A1 · 정형");
  tbl(s, [
    ["항목", "답 요지", "위치"],
    ["1. 최종 검증 코드와 같은가", "같음. produce 결과가 24회차 업로드본과 바이트 단위 일치, check ALL PASS. 전체 재학습 시 EC 최대 .013 차이(TabPFN GPU→CPU)", "정형 README ⑥"],
    ["4. 공개 리더보드 사용처", "제출물 비교·선택: 여러 부품을 제출 비교로 채택. 상수 결정: TabPFN 비중 0.4 한 건(0.3·0.4 두 후보 중 선택, 아래)", "정형 README ⑤"],
    ["4-보충. 0.3·0.4 고른 방법", "7회차(0.3)와 8회차(0.4)는 온도 열에서 이 비중만 다름 → 공개 온도 점수 .5193 대 .5160 비교로 0.4 유지. 두 후보는 검증으로 좁힘", "08회차 설명자료, 정형 README ⑤"],
    ["5. 외부 자료·사전학습·생성형 AI", "외부 데이터 없음. TabPFN v2(Prior Labs, 패키지 Apache-2.0). 코딩에 Claude·Codex", "정형 README ①"],
    ["6. AI가 한 것 / 팀이 한 것", "팀: 문제 정의·채택 기준·규정 경계·W2 제거와 현장형 정리·제출 선택 / AI: 코드·실험 실행·가설 제안·문서 초안", "본문 14쪽"],
  ], { x: 0.6, y: 1.6, w: 12.1, colW: [2.9, 6.9, 2.3], fontSize: 12 });
  s.addNotes("자기 점검 1: 같음. checks_v35e.py produce가 만드는 파일은 24회차 업로드본과 바이트 단위로 같고(SHA256 f1224ee1…), 26회차 업로드본(submission/)과는 줄바꿈만 다르고 숫자는 같음. check ALL PASS. 원본 CSV로 전체 재학습하면 EC 구성원·계절표·온도 NX는 저장본과 같고 HR5 학습 입력만 TabPFN GPU→CPU 차이로 최종 답안 온도 최대 .0016, EC 최대 .013 차이. [정형 README ⑥]\n\n자기 점검 4: 공개 리더보드는 (가) 제출물 비교·선택 — 로컬 기준을 다 채우지 못한 구성을 하나의 후보로 제출하고 공개 점수가 나아지면 다음 후보의 바탕으로 유지(정형 README ⑤ 표 '최종 구성에 들어간 경위' 열), (나) 상수 결정 — TabPFN 비중 0.4 한 건. 점수로 값을 계산하거나 평가 정답을 역산하지 않았습니다. 제출끼리 공개 점수 비교는 운영위원회 10/5 중간 안내 뒤 추가 안내에서 허용된 범위로 이해했습니다.\n\n(팀별 정형 1) TabPFN 0.3·0.4: 7회차 submission_09.csv(10-01)는 따뜻한 행 TabPFN 몫 0.3, 8회차 submission_11.csv(10-03)는 0.4. 두 제출 사이에 온도 열에서 바뀐 것은 이 비중 하나뿐(제출/08회차_2026-10-03(팀)/설명자료_11_온도W40G_EC계절v2.md 표). 두 제출의 공개 온도 점수 .5193 대 .5160을 비교해 0.4를 유지. 두 후보는 검증으로 좁힌 것 — 비중을 올릴수록 추운 날이 나빠져(0.6에서 뚜렷) 0.4를 위쪽 한계로 봤고, 0.3과 0.4는 검증 우열이 엇갈렸습니다.\n\n자기 점검 5: 외부 데이터 없음, 예측 단계 외부 조회·원격 추론 없음. 사전학습 TabPFN v2 회귀 가중치(Prior Labs, HF Prior-Labs/TabPFN-v2-reg, SHA256 2ab5a07d…). tabpfn 패키지 Apache-2.0(메타데이터 확인), 가중치는 HF 저장소 라이선스 표기를 따름(원문 대조 미실시). 코드 작성에 Claude·Codex, 예측에는 쓰지 않음. [정형 README ①]\n\n자기 점검 6(11-02 팀 확인): 팀이 판단·설계 — 문제 정의, 채택 기준(실행 전 고정, 모든 검증기 같은 방향), 규정 경계(예측 시점 이전 정보까지만), W2 제거와 현장형 정리, 제출 후보와 최종본 선택. AI(Claude·Codex)가 수행 — 코드 작성, 실험 실행, 가설·실험 제안, 문서 초안.");
}
// A2
{
  const s = appx("자기 점검 2·3 — 정형: 규칙성에 기대는 처리와 현장에서의 변화", "A2 · 정형 · 공통 1");
  tbl(s, [
    ["처리", "기대는 대회 자료 성질", "현장에서는"],
    ["FL 링커", "하루 기록이 섞이고 같은 동이 ID를 오감", "필요 없음 — 달력으로 어제가 정해짐"],
    ["어제 정답 앵커(23기록)", "평가 기록 사이에 정답 있는 학습 기록이 끼어 있음", "전날 배지 센서 실측이 있을 때만"],
    ["예측 사슬(17기록)", "앞선 평가 기록이 어제로 연결됨", "그대로 가능(어제의 내 예측)"],
    ["excl 특징", "학습 정답의 이어짐", "필요 없음"],
    ["계절 지표", "2차 구간 외기가 1차 구간 어떤 날과 거의 같음", "달력 날짜로 대체"],
    ["NX · HR5 · TX3", "해당 없음(그 시각까지 입력·예측만)", "그대로"],
  ], { x: 0.6, y: 1.6, w: 12.1, colW: [2.8, 5.0, 4.3], fontSize: 12 });
  T(s, bl(["2. 예측 시점 정보만: 평가 행은 같은 온실의 그 시각까지·앞 번호 기록만 — 절단 3곳·격리 2개 검사 ALL PASS", "3. 이 처리를 뺀 성능(ABL1, 11-02, 서술용): 학습 자료 검증기 2차 행 RMSE 온도 +3.1~+10.4%, EC +8.3~+30.5% · 계절 지표를 뺀 성능은 미실시", "학습 자료와 출처가 다른 자료: 예측 사슬 식은 다른 49개 온실에서 외부 확인(48곳 개선), 나머지 부품은 미실시"], 12), { x: 0.6, y: 5.0, w: 12.1, h: 1.8 });
  s.addNotes("자기 점검 2: 평가 행 특징은 평가 입력(test_X) 가운데 같은 온실의 그 시각까지 입력과 같은 온실 앞 번호 평가 기록의 입력만 씁니다. 뒤 시각·뒤 번호·다른 온실의 평가 입력은 어느 단계에서도 쓰지 않습니다. 학습 행 특징은 MASK 세계(test_X를 모두 비운 사본)에서 만듭니다. 학습 자료는 10/5 중간 안내에 따라 날짜 제한 없이 씁니다(평가 행보다 뒤 날짜의 학습 정답 포함). 검사: checks_v35e.py check(미래 입력 절단 3곳, 다른 온실 격리 2개) ALL PASS, check_stage0_causality.py(중간값 단계) ALL PASS, HR5 학습 입력은 test_X를 모두 비워도 같음(최대 차 5.0e-11). [정형 README ③]\n\n자기 점검 3 · 공통 1: 표의 처리들이 대회 자료 성질에 기댑니다. 링커 반사실 검사(10-20): 평가 기록을 후보에서 모두 빼도 정답 앵커 15기록의 어제는 그대로, 사슬 13기록은 정의상 0. excl 가중을 0으로 두면 어제 판단이 바뀐 4기록은 앵커·사슬 대상이 아니지만, 앵커 4개의 확신이 .8 아래로 내려감 — excl이 평가 쪽 기록을 간접적으로 가려낼 수 있다는 점을 밝힘. [정형 README ③]\nABL1 끄기 실험(11-02 추가, 계획 사전 고정 커밋 45ac69c5, 서술용): 일반 방법만(온도 NX + TX3, EC pre + HR5) → 최종, 2차 행 RMSE: 온도 EXT10 .8387 → .7599, EXT12 .9059 → .8791, DIAG10 .7388 → .7146 / EC DIAG10 .2522 → .2329, EL1 .2787 → .2135, P2LOO .1913 → .1723. 온도 몫 중 사슬 0.4~3.1%p. 검증기는 학습 기록을 가린 것(평가 자료 아님). 계절 지표는 모든 안에 남아 있어 그것까지 뺀 성능은 미실시, 실제 어제로 거는 현장형 사슬도 미실시. 결과 문서: 집/클로드/발표/ABL1_어제이어붙이기_끄기_결과_2026-11-02.md, 카탈로그 7.069.\n학습 자료와 출처가 다른 자료: 사슬 식 0.9·exp(−h/8)는 다른 49개 온실 학습 자료로 외부 확인(48곳 개선). 다른 49개 온실은 이 확인에만 썼고 최종 모델 학습에는 쓰지 않았습니다. 나머지 부품은 미실시. [정형 README ①·④]");
}
// A3
{
  const s = appx("자기 점검 1·4·5·6 — 이미지", "A3 · 이미지");
  tbl(s, [
    ["항목", "답 요지", "위치"],
    ["1. 최종 검증 코드와 같은가", "같음. 제출 ZIP의 추론 코드 그대로, 공개 21장 CPU 재추론 21/21(정순·역순 차 0)", "이미지 README ⑥·⑧"],
    ["4. 공개 리더보드 사용처", "제출물 비교·선택: 나단9와 나단11 중 최종 선택에 환산점수(23.82 → 25.59) 비교. 사전 규칙(≥ .98)과 다르게 정한 이유 명시. 상수·문턱 결정에는 안 씀", "이미지 README ⑤ 끝"],
    ["5. 외부 자료·사전학습·생성형 AI", "AI Hub 596(이용 승인), CLIP(MIT), SD1.5·SDXL(OpenRAIL), Qwen-Image-Edit(Apache-2.0), ChatGPT 이미지 — 상세 A6", "이미지 README ①, data/README_data.md"],
    ["6. AI가 한 것 / 팀이 한 것", "팀: 문제 정의·채택 기준·AI Hub 선택과 신청·제출 선택 / AI: 코드·실험 실행·가설 제안·GPT 이미지 생성 작업·문서 초안", "본문 14쪽"],
  ], { x: 0.6, y: 1.6, w: 12.1, colW: [2.9, 6.9, 2.3], fontSize: 12 });
  s.addNotes("자기 점검 1: 같음. code/submission_inference/는 최종 ZIP 안의 추론 코드 그대로. 팀원 노트북 CPU(fp32, 10-17)에서 보관 ZIP을 그대로 풀어 공개 21장 21/21, 정순·역순 결과 차 0. 서버 공식 self_check PASS, 순서 독립 OK. [이미지 README ⑥·⑧]\n\n자기 점검 4: 계수·문턱·가중은 리더보드로 고르지 않음. 회차마다 미리 정한 후보 하나를 내고 점수로 채택 여부만 비교. 최종 선택: 사전 규칙은 'LB Accuracy ≥ .98이면 후보 유지, 아니면 나단9로 되돌림'. R21에서 나단11은 Accuracy .97로 나단9와 같게 표시됐지만 플랫폼 환산점수가 23.82 → 25.59로 올랐고, 내부 검증에서도 GPT 가짜를 더 잡고 실제 오판이 늘지 않아 나단11 유지. 미리 만든 두 판 중 최종을 고르는 데 환산점수 비교를 썼고 모델 값은 바꾸지 않음. 이미지 미션에 정형 미션 질의 답변(제출끼리 비교 허용)과 같은 기준이 적용된다고 본 것은 팀의 해석. [이미지 README ⑤]\n\n자기 점검 5: 부록 A6(공통 3) 참조. 사전학습 가중치: OpenAI CLIP ViT-L/14@336px·ViT-B/16(MIT), open_clip 3.3.0 pretrained='openai' 시각부만 fp16 safetensors로 저장, 추가 학습 없음. 생성 모델(SD1.5·SDXL·Qwen)은 학습 자료 제작에만 쓰고 제출물에 넣지 않음. 라이선스 표기는 모델 카드 기준 일반 지식. [이미지 README ①·⑥, data/README_data.md 3절]\n\n자기 점검 6(11-02 팀 확인): 팀이 판단·설계 — 문제 정의, 채택 기준, AI Hub 데이터셋 선택과 이용 신청, 제출 후보와 최종본 선택. AI(Claude·Codex)가 수행 — 코드 작성, 실험 실행, 가설·실험 제안, GPT 이미지 생성 작업, 문서 초안. 추론에는 AI 도우미를 쓰지 않음.");
}
// A4
{
  const s = appx("자기 점검 2·3 — 이미지: 각 분기가 작동하는 입력과 작동하지 않을 때", "A4 · 이미지 · 공통 1 · 팀별 2");
  tbl(s, [
    ["분기", "작동하는 입력", "작동하지 않는 사진"],
    ["R1 색 지문 대조", "DB 1위 거리 3.5 < d1 < 20, gap > 8, p ≥ .25 → p를 .9 이상으로", "기본 판별 확률 그대로"],
    ["정밀 원본 대조", "dg1 ≤ 1.011, gap ≥ 3.474(원본이 DB에 있음). 국소 차이 크면 가짜 확정(≥ .95), 거의 같고 p < .8이면 진짜 확정(≤ .3)", "이 단계를 건너뜀 → 기본 판별 + 전문가만"],
    ["형식 게이트", "JPEG · 4:2:0 · PIL Q92 양자화표와 완전히 같음 → 전문가 H·E_gr·E_gpt 적용", "세 전문가 미적용 → 기본 + R1 + E_pg·E_ps·E_L + 정밀 대조"],
  ], { x: 0.6, y: 1.6, w: 12.1, colW: [2.2, 6.0, 3.9], fontSize: 12 });
  T(s, bl(["2. 이미지 한 장만 입력, 이미지끼리 정보 교환 없음, 순서 독립 검사 OK", "3. 현장 사진은 대부분 DB에 원본이 없어 원본 대조가 거의 작동하지 않음 → 지문 DB 끈 성능: 균형 정확도 −1.4~−2.3%p(주로 실제 오탐 증가)", "형식 게이트를 끈 성능: 미실시 · 학습 자료와 출처가 다른 자료로 확인: 미실시"], 12), { x: 0.6, y: 4.6, w: 12.1, h: 2.1 });
  s.addNotes("자기 점검 2: 각 이미지는 그 파일 하나만 입력으로 계산. 경로를 정렬해 8장씩 GPU로 계산하므로 입력 순서와 무관하지만, 묶음 구성이나 GPU fp16/CPU fp32 차이로 확률에 작은 수치 차(나단3·4 사이 최대 .008, 판정 1장)가 생길 수 있음. 비공개 평가 자료는 학습·설정 선택에 쓰지 않음. [이미지 README ③]\n\n(팀별 이미지 2) 각 분기의 작동 조건과 미작동 시 처리 — 표 참조. 정의: d1 = 사진 12×16 RGB 평균과 DB의 평균 절대차 1위, gap = 2위 − 1위. 정밀 지문 = 덮게 축소·가운데 768×1024 크롭 후 흑백 24×32 평균 g와 64px 구역 12×16 라플라시안 분산 log e. dg1 = 평균 abs(g − g_db), lg·le = 국소 차이(상위 2%·5% 평균). 가짜 확정: lg > 6.569 또는 le > 52.95 → p ← max(p, .95). 진짜 확정: lg ≤ 3.933 & le ≤ 24.24 & |ge| ≤ 13 & p < .8 → p ← min(p, .3). 형식 게이트: PIL로 연 파일이 JPEG이고 서브샘플링 4:2:0이며 양자화표가 quality=92 표와 완전히 같을 때만 참. 예외 대체: 저수준 특징 실패 → p_CLIP만, E_f·E_gr·E_gpt 미적용 / CLIP 실패 → p_low만 / 원본 대조 실패 → R1·정밀 대조 미적용. [이미지 README ⑤ 표·정의]\n\n자기 점검 3 · 공통 1: 원본 대조는 '공개 샘플과 비슷한 외부 데이터셋에 원본이 있다'는 대회 자료 구성에 기댑니다. 현장 사진은 대부분 DB에 원본이 없으므로 지문 DB를 끈 판에 가까워집니다. 끄기 실험(10-20, 계획 사전 고정): R2 .964 → .945, R3 .962 → .948, R4 .947 → .924, R5 .944 → .921, 실제 오탐 R5 1.4% → 5.1%, 처리 변형 webp 37.5% → 75.8%, 가짜 탐지는 부분 생성 82.3 → 79.2%로 작음. 형식 게이트는 배포 형식(JPEG Q92 4:2:0)에 맞춘 것으로, 현장 사진 형식이 다르면 전문가 셋이 빠짐 — 끈 성능은 미실시. 학습 자료와 출처가 다른 자료로 확인: 미실시(봉인 R2~R5도 같은 AI Hub 데이터셋에서 촬영 묶음만 다르게 뗀 것, 공개 21장은 운영진 자료지만 설계·선택에 썼음).");
}
// A5
{
  const s = appx("성능 수치의 출처 — 어떤 자료로 냈고, 선택에도 썼는가", "A5 · 공통 2");
  tbl(s, [
    ["수치", "낸 자료", "선택에도 썼나", "학습과 겹침"],
    ["정형 공개 점수(.4608 · .1148)", "플랫폼 평가 자료", "예 — 제출 비교로 부품 채택, TabPFN 0.4", "없음"],
    ["정형 검증기(DIAG10·EL1·P2LOO·EXT10·EXT12)", "F13·F47 학습 기록을 기록 단위로 가림", "예 — 채택 판정, 일부 값 선택(HR5 λ, EC 앵커 강도 등)", "같은 학습 자료 안에서 나눔, 링커 가중도 같은 정답으로 맞춤"],
    ["정형 ABL1 끄기 결과", "위 검증기", "아니오(서술용)", "위와 같음"],
    ["이미지 공개 점수(.97 · 25.59)", "운영진 시험 이미지", "예 — 회차별 채택, 나단11 최종 선택", "없음"],
    ["이미지 봉인 R2~R5", "AI Hub 596, 촬영 묶음이 안 겹치게 뗀 실제·가짜", "예 — 후보 채택 판정", "같은 데이터셋·같은 제작 방법, 원본 사진은 안 겹침"],
    ["이미지 공개 21장", "운영진 공개 샘플", "예 — 초기 구성 선택 5회 이상, 이후 21/21 유지 점검", "학습에는 안 넣음"],
    ["이미지 문턱 θ", "실제 풀(학습 실제 OOF + 미사용 실제)", "예 — 문턱 자체", "학습 실제와 겹침(OOF)"],
  ], { x: 0.6, y: 1.6, w: 12.1, colW: [3.2, 3.1, 3.2, 2.6], fontSize: 11 });
  s.addNotes("공통 2 — 발표에서 제시한 수치마다 낸 자료와 선택 사용 여부.\n- 정형 공개 점수: 플랫폼 평가. 제출 비교로 부품 채택에 사용(정형 README ⑤ 표).\n- 정형 검증기: F13·F47 학습 기록 400일을 기록 단위로 가린 것. DIAG10 = 10겹 무작위, EL1·P2LOO = 2차 구간 기록 묶음·한 기록씩, EXT10·EXT12 = 하루 최저 10·12℃ 미만인 날만 가림. 채택 판정에 썼고, 일부 값은 이 검증기에서 선택(HR5 λ는 DIAG10에서 판정 전 선택, EC 앵커 강도 0.8·p는 90조합 중 선택, TAG2·CHT 선택지). 같은 학습 자료 안에서 기록을 나누며, 같은 기록에서 나온 행은 같은 폴드에 둠. 링커 가중은 같은 학습 정답으로 맞춤.\n- 발표 5쪽 '+67~84%'(STK1)와 '78~92%'(BME1)도 이 검증기 수치(카탈로그 7.014·6.756), 선택에는 각각 기각·설계 근거로 사용.\n- 정형 ABL1: 위 검증기, 서술용(선택에 쓰지 않음).\n- 이미지 공개 점수: 운영진 시험 이미지(코드 제출 방식이라 팀은 접근 불가). 회차별 채택과 나단11 최종 선택에 사용.\n- 이미지 봉인 R2~R5: 학습에 쓰지 않은 AI Hub 실제·가짜, 촬영 묶음이 학습과 겹치지 않음. 학습·문턱 계산에는 쓰지 않고 후보 채택 판정 때만 열었으며 10-20 지문 DB 끄기 실험에서 다시 씀. 같은 데이터셋·같은 제작 방법이라 출처가 다른 자료는 아님.\n- 이미지 공개 21장: 학습 미사용. 10-10~12 초기 후보 구성(가짜 제작 5판 포함 여부, 기본 판별기 조합) 선택에 5회 이상 지표로 썼고, 이후 21/21 유지를 비열화 점검 조건으로 씀. [이미지 data/README_data.md 4절]\n- 이미지 문턱 θ: 실제 풀(학습 실제 묶음 5폴드 OOF + 미사용 실제 1,395)에서 실제 0.25%만 뒤집는 분위. [이미지 README ⑤]");
}
// A6 (2장)
{
  const s = appx("이미지 자료의 외부 처리와 이용조건 ①·②", "A6 · 이미지 · 공통 3");
  box(s, 0.6, 1.6, 5.9, 5.1, C.background2);
  T(s, "① 외부 생성형 서비스에 넣은 공개 데이터", { x: 0.85, y: 1.75, w: 5.4, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
  T(s, bl(["서비스: ChatGPT 내장 이미지 생성·편집(OpenAI Codex 도우미로 수행)", "데이터셋: AI Hub 596", "입력: 서로 다른 AI Hub 원본 525장(gpt1 75 · gpt2 150 · gpt3 150 · gptr 150, 겹침 0)", "결과 600장(25장은 원본 없이 문장만으로)", "최종 학습: E_gpt에 600장 전부, E_pg·E_ps·E_L에 gpt1·gpt2 250장"], 13), { x: 0.85, y: 2.25, w: 5.4, h: 4.3 });
  tbl(s, [
    ["환경", "단계", "처리·보관 국가"],
    ["연구실 GPU 서버(ADILAB)", "가짜 생성 · 특징 계산 · 학습", "국내(학교 안)"],
    ["연구실 PC", "AI Hub tar 추출 · 프로그램 가짜 · 저수준 특징", "국내"],
    ["Google Colab", "정밀 지문 DB 제작 · 추론 시간 측정", "미확인"],
    ["Google Drive(farmai_sync)", "AI Hub 원본(고른 사진)·GPT 세트·tar별 지문 파일 보관(장소 간 동기화)", "미확인"],
  ], { x: 6.8, y: 1.6, w: 5.9, colW: [1.9, 2.6, 1.4], fontSize: 11 });
  T(s, "② 원격 환경에서의 처리·보관", { x: 6.8, y: 5.6, w: 5.9, h: 0.4, fontSize: 13, bold: true, color: C.accent5 });
  s.addNotes("공통 3 ①: ChatGPT 내장 이미지 생성·편집 기능에 AI Hub 「지능형 수직농장 통합 데이터(딸기)」(596) 원본 사진을 넣어 편집본을 만들었습니다(작업은 OpenAI Codex 도우미로 수행). 입력한 AI Hub 원본은 서로 다른 525장 — gpt1 75장(전체 스타일·부분 생성·부분 스타일 각 25), gpt2 150장, gpt3 150장, gptr 150장, 묶음 사이 겹침 0. 결과 이미지는 600장: gpt1 100(전체 생성 25장은 원본 없이 문장만으로), gpt2 150, gpt3 200(그중 '전체 다시 그리기' 50장은 같은 원본 50장을 편집하는 과정의 1단계 결과물), gptr 150. 최종 학습: GPT 편집 전문가 E_gpt에 600장 전부, 전문가 E_pg·E_ps·E_L에 gpt1·gpt2 250장. 근거: Drive lab_claude_local/im_set_gpt1~3/manifest.csv의 source 열, 제출/최종검증/팜모니_이미지/data/gpt_generation/codex_gpt_regen_20261015/목록.csv(11-02 검산). 참고: 최종 검증 자료의 gpt_generation/codex_ai_gen_work_20261014_v3/manifest_v3.csv(150행)에는 1단계 결과물 50장이 빠져 있습니다.\n\n공통 3 ②: 연구실 GPU 서버(ADILAB, RTX 6000 Ada) — SD·SDXL·Qwen 생성, CLIP 특징, 학습, 생성·학습용 이미지 세트 보관. 학교 안 서버, 국내(팀 확인). 연구실 PC(Windows) — AI Hub tar 추출, 프로그램 가짜 제작, 저수준 특징. Google Colab — 정밀 지문 DB 제작(colab_정밀지문.py), 추론 시간 측정(521장 대용품). 처리 국가 미확인. Google Drive(farmai_sync, 팀원 개인 계정) — 연구실 PC와 집 노트북 사이 동기화용으로 AI Hub에서 고른 원본 사진(im_real_raw 등), GPT 세트, tar별 지문 파일을 보관(11-02 폴더 확인). 보관 국가 미확인.");
}
{
  const s = appx("이미지 자료의 외부 처리와 이용조건 ③·④ — 지문 DB 두 개", "A6 · 이미지 · 공통 3 · 팀별 3");
  tbl(s, [
    ["DB", "사진별로 담은 값", "크기", "원천·장수"],
    ["fp_db.npz.xz", "12×16 컬러 평균(아주 작은 컬러 축소본)", "576값(사진당 576바이트)", "AI Hub 596 원천 152,043장 전체"],
    ["finefp_db.npz.xz", "24×32 흑백 평균(아주 작은 흑백 축소본) + 12×16 라플라시안 분산 log", "768 + 192값", "같은 152,043장"],
  ], { x: 0.6, y: 1.6, w: 12.1, colW: [2.0, 4.9, 2.4, 2.8], fontSize: 12 });
  box(s, 0.6, 3.55, 5.9, 3.15, C.background2);
  T(s, "③ 이용조건 확인·판단", { x: 0.85, y: 3.7, w: 5.4, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
  T(s, bl(["AI Hub: 이용 신청·승인. 재배포 제한은 일반 지식 수준으로 판단, 약관 원문 대조는 미실시", "그래서 10-20 AI Hub 소재 16,197장을 파일 대신 원본 대응 해시로 제출 — 그 밖에 바꾼 결정은 없음", "생성 모델·ChatGPT: 모델 카드·이용약관 기준 일반 지식", "출처를 확인하지 못한 자료: 해당 없음"], 12), { x: 0.85, y: 4.15, w: 5.4, h: 2.5 });
  box(s, 6.8, 3.55, 5.9, 3.15, C.text2);
  T(s, "④ 제출 모델에 넣은 참조 자료", { x: 7.05, y: 3.7, w: 5.4, h: 0.4, fontSize: 15, bold: true, color: C.accent4 });
  T(s, bl(["위 두 DB는 사진별 아주 작은 축소본 성격(원 사진 768×1024 대비 화소 수 1/4,096 · 1/1,024)", "원 사진을 대신할 해상도는 아니지만 사진별 값이므로 대상으로 보고 밝힘", "넣은 판단: 출처를 밝히고 포함(팀 결정 10-12), AI Hub 약관 원문 대조는 미실시"], 12, C.background1), { x: 7.05, y: 4.15, w: 5.4, h: 2.5 });
  s.addNotes("공통 3 ③: AI Hub 데이터는 이용 신청·승인 후 팀원 계정으로 원천 tar 306개를 받았습니다. 승인 이용자만 쓸 수 있고 원본 재배포가 제한된다는 판단은 AI Hub 이용 약관에 대한 일반 지식 수준이며 원문 대조는 하지 않았습니다(미실시). 이에 따라 10-20 최종 검증 자료에서 AI Hub 소재 16,197장(원본과 그 편집·합성본)은 파일로 넣지 않고 원본 대응 해시 표(data/source_sha256.csv)와 문서로 대신했고, 원본 없이 문장만으로 만든 967장만 파일로 냈습니다. 이용조건 때문에 바꾸거나 쓰지 않기로 한 결정은 이것 하나입니다(11-02 팀 확인). 생성 모델 라이선스(SD1.5 CreativeML OpenRAIL-M, SDXL OpenRAIL++-M, sdxl-vae-fp16-fix MIT, Qwen-Image-Edit Apache-2.0)와 ChatGPT 이용약관(생성물 권리 이용자 귀속)은 모델 카드·약관에 대한 일반 지식으로 적었습니다. 출처를 확인하지 못한 자료: 해당 없음 — 모든 학습 이미지에 원본 경로(manifest의 source)나 생성 기록이 있습니다. [이미지 data/README_data.md 1·3절]\n\n공통 3 ④ · 팀별 이미지 3: 제출 모델(model/)에 지문 DB 두 개를 넣었습니다. fp_db.npz.xz — AI Hub 596 원천 사진 152,043장 전체 각각의 12×16 RGB 평균(576값, 사진당 576바이트), im11_fingerprint_db.py로 만들고 나단7에서 lzma 재압축. finefp_db.npz.xz — 같은 152,043장 각각을 덮게 축소·가운데 768×1024 크롭한 뒤 흑백 24×32 평균(768값)과 64px 구역 12×16 라플라시안 분산 log값(192값), Colab colab_정밀지문.py(+_v2)로 제작. 두 값은 사실상 사진의 아주 작은 축소본(12×16 컬러, 24×32 흑백)이므로 이를 밝힙니다(11-02 팀 결정). 원 사진 대비 화소 수는 각각 1/4,096, 1/1,024입니다. 제출 모델에 넣을 때의 판단: 사진이 아니라 사진별 숫자 통계라고 보고 출처를 밝혀 포함(팀 결정 2026-10-12). AI Hub 약관 조항과의 원문 대조는 미실시. 학습으로 얻은 모델 가중치는 이 항목에 해당하지 않습니다. [이미지 README ①·④·⑧, data/README_data.md 1절]");
}
// A7 제출 이력
{
  const s = appx("정형 제출 이력 26회 (공개 점수, 낮을수록 좋음)", "A7 · 참고");
  const H = [
    [1, "09-22", ".7450", ".2442", "기준 모델"], [2, "09-25", ".6666", ".2287", "특징 축소·모델 혼합"], [3, "09-26", ".5624", ".2055", "물리 기준선"], [4, "09-26", ".5697", ".2055", "한랭 보정(악화)"], [5, "09-26", ".5456", ".2134", "학습 행 가중 0.2"],
    [6, "09-30", ".5251", ".2055", "실내온도 게이트 섞기"], [7, "10-01", ".5193", ".2774", "TabPFN 0.3"], [8, "10-03", ".5160", ".1455", "TabPFN 0.4 · EC 계절"], [9, "10-06", ".5127", ".1384", "day → 계절 지표"], [10, "10-11", ".5126", ".3538", "EC 고EC 날 제외(악화)"],
    [11, "10-11", ".5089", ".1305", "새벽 보정 · W2 포함"], [12, "10-11", ".5012", ".1271", "어제 앵커 첫 판 · W2"], [13, "10-12", ".4883", ".1279", "앵커 확장 · W2"],
  ];
  const H2 = [
    [14, "10-13", ".4864", ".1077", "일치 앵커 · W2"], [15, "10-14", ".4853", ".0979", "HR5 · W2"], [16, "10-15", ".4861", ".1225", "W2 등 제거"], [17, "10-15", ".4817", ".1170", "현장형 구성"], [18, "10-15", ".4922", ".1173", "GRU 30%(악화)"],
    [19, "10-16", ".4706", ".1153", "예측 사슬"], [20, "10-16", ".4754", ".1153", "이력 재연결(악화)"], [21, "10-16", ".4685", ".1153", "사슬 문턱 .5"], [22, "10-16", ".5000", ".1153", "범위 밖 보정(악화)"], [23, "10-16", ".4665", ".1153", "TAG2 앵커"],
    [24, "10-16", ".4608", ".1148", "CHT · EC TAG2 (최종)"], [25, "10-16", ".4845", ".1148", "실험, 되돌림"], [26, "10-16", ".4608", ".1148", "24회차 재제출"],
  ];
  const mk = (rows) => [["회", "날짜", "온도", "EC", "바꾼 것"], ...rows.map((r) => r.map(String))];
  tbl(s, mk(H), { x: 0.6, y: 1.55, w: 5.95, colW: [0.5, 0.8, 0.75, 0.75, 3.15], fontSize: 10, rowH: 0.36 });
  tbl(s, mk(H2), { x: 6.75, y: 1.55, w: 5.95, colW: [0.5, 0.8, 0.75, 0.75, 3.15], fontSize: 10, rowH: 0.36 });
  s.addNotes("출처: 제출/제출이력_정형_2026-10-20.md(팀 회차 = 플랫폼 회차, 팀원 확인 10-20). 25·26회차는 마감 직전 운영진이 제출 기회를 1회 더 준 데 따른 것. 6회차는 팀원 개인 몫 제출. W2 규칙(같은 온실 앞선 일차 기록이 평가 자료에 있는지로 갈리는 EC 규칙)은 11~15회차에 들어 있었고 주최 측 10/16 답변에서 금지 대상으로 확인됨. 팀은 그 전인 16회차(10-15)부터 이 규칙을 뺐고, 이후 최종 26회차까지 모든 제출에 없음. 제출마다 미리 만든 후보 구성 하나를 냈고, 공개 점수로는 다음 후보의 바탕으로 유지할지만 정했습니다.");
}
// A8 부품 채택 경위
{
  const s = appx("정형 부품별 값을 정한 방법과 최종 구성에 들어간 경위", "A8 · 참고");
  tbl(s, [
    ["부품·값", "값을 정한 방법", "로컬 판정", "들어간 경위"],
    ["NX 섞기 0.4/0.2/0.4, 게이트 8~10℃", "검증 + 두 후보 제출 비교", "DIAG10 −0.8%, 추운 검증 +0.1~0.2%(미달)", "제출 비교(0.3 대 0.4)"],
    ["CODEX·TabPFN day → 계절 지표", "검증 탐색", "2차 DIAG10 −1.35%(탐색)", "제출 비교"],
    ["day·second 제거", "검증기 4종 비교", "혼재", "현장형 정리로 제출 비교"],
    ["FL 링커·어제 앵커", "학습 정답으로 적합", "이전 판 P .093(미달)", "제출 비교"],
    ["TAG2 앵커 확장", "검증 블록에서 데이터 선택", "우월 P .12, 대체 통과", "제출 비교"],
    ["TX3 0.25 / 0.8 / 10℃", "학습 정답 측정값", "통과(44칸, P .0047)", "로컬 채택"],
    ["예측 사슬 0.9·exp(−h/8)", "검증 사슬 행 적합", "통과(P .0003)", "로컬 채택"],
    ["사슬 문턱 p ≥ .5", "확신 구간별 진짜 연결 비율", "통과(P .0029)", "로컬 채택"],
    ["CHT 사슬 확장", "검증 블록에서 선택", "대체 통과, 손해 0", "제출 비교"],
    ["HR5 λ 0.5", "DIAG10에서 판정 전 선택", "P2LOO −9.1%, EL1 −4.1%", "제출 비교"],
    ["EC 앵커 0.8·p", "검증 90조합 중 선택", "P .35(미달), 6/6 개선", "제출 비교"],
    ["EC TAG2 앵커", "TAG2와 같은 조건", "P .0026, DIAG10 한 기록 손해(미달)", "제출 비교"],
  ], { x: 0.6, y: 1.55, w: 12.1, colW: [3.2, 3.0, 3.6, 2.3], fontSize: 10, rowH: 0.38 });
  s.addNotes("출처: 정형 README ⑤ 표. '로컬 판정' = 실행 전에 고정한 채택 기준(모든 검증기·시드 같은 방향 개선 + 블록 부트스트랩 P < .025)으로 검증기 결과를 판정한 것. '대체 판정' = 기존 부품을 대신할 만큼 손해가 없는지를 보는 약한 기준. '제출 비교' = 로컬 기준을 다 채우지 못한 구성을 하나의 후보로 제출하고 공개 점수가 직전 최고보다 나아져 다음 구성의 바탕으로 유지한 경우. EC 구성원·shrink·0.8/0.2는 9회차 EC 구성(판정표 통과), EC의 SG2·W2·B5 제거는 대회 자료 표식·구조 규칙 제거로 현장형 구성(16·17회차)에서 제출 비교로 채택. 근거 기록은 각 스크립트 머리 주석과 데이터 단서 카탈로그.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
