// 발표 템플릿 시안 5종 v2 (2026-11-04, 연구실 클로드) — v1 + 사용자: 팀 이름 '팜모니', 글꼴 Gmarket Sans TTF(굵은 글씨는 Bold, 나머지는 Medium).
// (v1) 발표 템플릿 시안 5종 (2026-11-04, 연구실 클로드) — 사용자 요청: "AI티 안 나게" 고를 수 있게 5개.
// 각 시안은 표지 1장 + 같은 본문 장(1.4 온도 기본 예측) 1장. 내용은 G05v16 그대로, 모양만 다름.
// 피한 것: 그라데이션, 둥근 카드·그림자, 아이콘·이모지, 왼쪽 강조 막대, 보라·남색 조합, 큰 숫자 타일.
// 실행: NODE_PATH=<node_modules> node build_template_samples_v1.js
const pptxgen = require("pptxgenjs");
const path = require("path");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 × 7.5
pres.title = "템플릿 시안 5종";

const BOLD = "Gmarket Sans TTF Bold", MED = "Gmarket Sans TTF Medium";
const SERIF = BOLD, SANS = MED, MG = MED;
// 굵게 표시한 글은 Bold 글꼴로 바꾸고 가짜 굵기는 끔
const gm = (o) => (o.bold || o.fontFace === BOLD ? Object.assign(o, { fontFace: BOLD, bold: false }) : o);
const TITLE = "현장에서 그 시각에 알 수 있는 정보로\n예측하고 판별하기";
const LEAD = "기본 예측은 세 모델을 실내 온도에 따라 섞는다";
const MODELS = [
  ["A", "실내외 센서와 난방·환기 기록, 앞선 며칠의 변화까지 함께 봄"],
  ["B", "오늘 0시부터 지금까지 센서와 구동기가\n어떻게 변했는지 봄"],
  ["C", "표 자료로 미리 학습한 모델에 기존 기록을\n예시로 주고 지금의 배지 온도를 추정"],
];
const W = [["10℃ 이상", "40%", "20%", "40%"], ["9℃ 예시", "40%", "40%", "20%"], ["8℃ 이하", "40%", "60%", "0%"]];
const CAP4 = "표 4. 온도 기본 예측에 쓰는 세 모델. A·B는 물리 기준 예측에 학습한 보정을 더함";
const CAP5 = "표 5. 실내 온도에 따른 모델 비중";
const NOTE5 = "8~10℃에서는 C의 비중을 0%에서 40%까지\n서서히 늘리고, B의 비중은 그만큼 줄인다.";
const FOOT = "섞는 식: 정형 README ②(온도 1). 비중을 고른 경위: 정형 README ⑤, 부록 A7.";
const MEMBERS = "조나단(발표) · 박찬유 · 반지원 · 김태경 · 김우진";

const T = (s, text, o) => s.addText(text, gm(Object.assign({ margin: 0, valign: "top", isTextBox: true }, o)));
const line = (s, x, y, w, c, wt) => s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color: c, width: wt } });
const vline = (s, x, y, h, c, wt) => s.addShape(pres.shapes.LINE, { x, y, w: 0, h, line: { color: c, width: wt } });
const rect = (s, x, y, w, h, fill) => s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { type: "none" } });
const none = { type: "none" };
const bd = (c, pt) => ({ type: "solid", color: c, pt });
// 표 하나를 스타일 함수로: cell(i, j, last) → options
function tbl(s, rows, cell, o) {
  const last = rows.length - 1;
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: gm(cell(i, j, last)) }))), Object.assign({ valign: "middle", margin: [5, 7, 5, 7] }, o));
}
const label = (no, name) => {
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  s.addNotes(`시안 ${no}: ${name}`);
  return s;
};

// ───────── 시안 1. 학술지형(지금 덱을 더 덜어냄) ─────────
// 흰 바탕, 명조 제목, 논문식 3선 표, 색은 거의 없음(먹색 + 올리브 한 점).
{
  const ink = "222222", body = "3A3A3A", mute = "8A8A8A", acc = "5F6B2E";
  let s = label(1, "학술지형");
  T(s, "제1회 전국 농과계 대학 농업 AI 경진대회 · 현장 발표", { x: 1.0, y: 1.4, w: 11, h: 0.4, fontFace: SANS, fontSize: 14, color: mute });
  T(s, TITLE, { x: 1.0, y: 2.0, w: 11.3, h: 1.9, fontFace: SERIF, fontSize: 38, bold: true, color: ink });
  line(s, 1.0, 4.35, 2.2, ink, 1.25);
  T(s, "팜모니", { x: 1.0, y: 4.6, w: 6, h: 0.45, fontFace: SERIF, fontSize: 19, bold: true, color: acc });
  T(s, MEMBERS + "\n2026년 11월 6일, 대구 EXCO", { x: 1.0, y: 5.1, w: 11, h: 0.8, fontFace: SANS, fontSize: 14, color: body, lineSpacingMultiple: 1.3 });

  s = label(1, "학술지형");
  T(s, "1.4  온도 기본 예측", { x: 0.8, y: 0.35, w: 8, h: 0.3, fontFace: SANS, fontSize: 12, color: mute });
  T(s, LEAD, { x: 0.8, y: 0.68, w: 11.8, h: 0.7, fontFace: SERIF, fontSize: 26, bold: true, color: ink });
  const c1 = (i, j, last) => ({ fontFace: SANS, fontSize: 15, bold: i === 0, color: i === 0 ? ink : body,
    border: [i === 0 ? bd(ink, 1.5) : none, none, i === 0 ? bd(ink, 0.75) : i === last ? bd(ink, 1.5) : bd("E4E4E4", 0.5), none] });
  tbl(s, [["모델", "무엇을 보나"], ...MODELS], c1, { x: 0.9, y: 1.95, w: 6.9, colW: [0.9, 6.0], rowH: [0.5, 0.8, 0.8, 0.8] });
  T(s, CAP4, { x: 0.9, y: 5.05, w: 6.9, h: 0.6, fontFace: SANS, fontSize: 12, color: mute });
  tbl(s, [["실내 온도", "A", "B", "C"], ...W], (i, j, l) => Object.assign(c1(i, j, l), { align: j ? "center" : "left", fontSize: 16 }), { x: 8.25, y: 1.95, w: 4.35, colW: [1.5, 0.95, 0.95, 0.95], rowH: 0.6 });
  T(s, CAP5, { x: 8.25, y: 4.45, w: 4.35, h: 0.35, fontFace: SANS, fontSize: 12, color: mute });
  T(s, NOTE5, { x: 8.25, y: 4.95, w: 4.35, h: 0.95, fontFace: SANS, fontSize: 15, color: body });
  T(s, FOOT, { x: 0.8, y: 6.95, w: 10.5, h: 0.3, fontFace: SANS, fontSize: 9, color: mute });
  T(s, "6", { x: 11.9, y: 6.95, w: 0.7, h: 0.3, fontFace: SANS, fontSize: 10, color: mute, align: "right" });
}

// ───────── 시안 2. 연구보고서형(국내 연구기관 보고서 문법) ─────────
// 위쪽 진한 가로줄 + 장 번호, □·○ 글머리, 표 머리만 옅은 회색 칸, 바깥 테두리 실선.
{
  const ink = "1F2A1F", body = "333333", mute = "7A7A7A", acc = "2F5233", head = "EDEFEA";
  let s = label(2, "연구보고서형");
  rect(s, 0, 0, 13.33, 0.18, acc);
  T(s, "제1회 전국 농과계 대학 농업 AI 경진대회  |  현장 발표", { x: 0.9, y: 0.55, w: 11, h: 0.35, fontFace: MG, fontSize: 12, color: mute });
  T(s, TITLE, { x: 0.9, y: 2.2, w: 11.5, h: 1.7, fontFace: MG, fontSize: 36, bold: true, color: ink });
  T(s, "정형: 시설 딸기 배지 온도·EC 예측  /  이미지: 딸기 사진의 실제·가짜 판별", { x: 0.9, y: 4.0, w: 11.5, h: 0.4, fontFace: MG, fontSize: 16, color: body });
  tbl(s, [["팀", "팜모니"], ["구성", MEMBERS], ["일시", "2026년 11월 6일, 대구 EXCO"]],
    (i, j) => ({ fontFace: MG, fontSize: 13, color: j ? body : ink, bold: !j, fill: { color: j ? "FFFFFF" : head }, border: [bd("BFC4BA", 0.75), bd("BFC4BA", 0.75), bd("BFC4BA", 0.75), bd("BFC4BA", 0.75)] }),
    { x: 0.9, y: 5.0, w: 7.2, colW: [1.2, 6.0], rowH: 0.42 });

  s = label(2, "연구보고서형");
  line(s, 0.7, 1.25, 11.93, acc, 2.25);
  T(s, "1-4", { x: 0.7, y: 0.5, w: 1.0, h: 0.6, fontFace: MG, fontSize: 20, bold: true, color: acc, valign: "bottom" });
  T(s, "온도 기본 예측", { x: 1.75, y: 0.5, w: 8, h: 0.6, fontFace: MG, fontSize: 24, bold: true, color: ink, valign: "bottom" });
  T(s, "□ " + LEAD, { x: 0.8, y: 1.5, w: 11.8, h: 0.45, fontFace: MG, fontSize: 18, bold: true, color: ink });
  const c2 = (i) => ({ fontFace: MG, fontSize: 14, bold: i === 0, color: i === 0 ? ink : body, fill: { color: i === 0 ? head : "FFFFFF" }, border: [bd("A9AFA3", 0.75), bd("A9AFA3", 0.75), bd("A9AFA3", 0.75), bd("A9AFA3", 0.75)] });
  tbl(s, [["모델", "무엇을 보나"], ...MODELS], c2, { x: 0.9, y: 2.2, w: 6.9, colW: [0.9, 6.0], rowH: [0.45, 0.8, 0.8, 0.8] });
  T(s, "<" + CAP4.replace("표 4. ", "표 4> "), { x: 0.9, y: 5.2, w: 6.9, h: 0.6, fontFace: MG, fontSize: 11, color: mute });
  tbl(s, [["실내 온도", "A", "B", "C"], ...W], (i, j) => Object.assign(c2(i), { align: j ? "center" : "left", fontSize: 15 }), { x: 8.25, y: 2.2, w: 4.35, colW: [1.5, 0.95, 0.95, 0.95], rowH: 0.55 });
  T(s, "<표 5> 실내 온도에 따른 모델 비중", { x: 8.25, y: 4.5, w: 4.35, h: 0.35, fontFace: MG, fontSize: 11, color: mute });
  T(s, "○ 8~10℃에서는 C의 비중을 0%에서 40%까지\n   서서히 늘리고, B의 비중은 그만큼 줄인다.", { x: 8.25, y: 4.95, w: 4.35, h: 0.95, fontFace: MG, fontSize: 14, color: body });
  line(s, 0.7, 6.85, 11.93, "C8CCC4", 0.5);
  T(s, "※ " + FOOT, { x: 0.7, y: 6.95, w: 10.5, h: 0.3, fontFace: MG, fontSize: 9, color: mute });
  T(s, "- 6 -", { x: 11.6, y: 6.95, w: 1.0, h: 0.3, fontFace: MG, fontSize: 10, color: mute, align: "right" });
}

// ───────── 시안 3. 세미나 노트형(연구실 발표) ─────────
// 왼쪽 좁은 세로 칸에 장 정보, 오른쪽 넓은 본문. 제목은 작고 굵게, 회색 위주, 표는 줄 없이 간격으로.
{
  const ink = "1E1E1E", body = "404040", mute = "939393", acc = "8C5A1E";
  let s = label(3, "세미나 노트형");
  vline(s, 3.1, 0.9, 5.7, "D0D0D0", 0.75);
  T(s, "팜모니\n\n2026. 11. 6.\n대구 EXCO", { x: 0.8, y: 0.95, w: 2.1, h: 2.2, fontFace: SANS, fontSize: 13, color: mute, lineSpacingMultiple: 1.2 });
  T(s, "현장 발표", { x: 0.8, y: 6.2, w: 2.1, h: 0.35, fontFace: SANS, fontSize: 12, color: acc, bold: true });
  T(s, TITLE, { x: 3.6, y: 2.4, w: 9.0, h: 1.7, fontFace: SANS, fontSize: 34, bold: true, color: ink });
  T(s, "정형: 시설 딸기 배지 온도·EC 예측\n이미지: 딸기 사진의 실제·가짜 판별", { x: 3.6, y: 4.3, w: 9, h: 0.8, fontFace: SANS, fontSize: 16, color: body, lineSpacingMultiple: 1.2 });
  T(s, MEMBERS, { x: 3.6, y: 6.2, w: 9, h: 0.35, fontFace: SANS, fontSize: 13, color: mute });

  s = label(3, "세미나 노트형");
  vline(s, 3.1, 0.6, 6.2, "D0D0D0", 0.75);
  T(s, "1.4\n온도 기본 예측", { x: 0.8, y: 0.62, w: 2.1, h: 0.9, fontFace: SANS, fontSize: 13, color: mute, lineSpacingMultiple: 1.2 });
  T(s, "식: 정형 README ②\n경위: README ⑤, 부록 A7", { x: 0.8, y: 5.95, w: 2.2, h: 0.7, fontFace: SANS, fontSize: 9.5, color: mute });
  T(s, LEAD, { x: 3.6, y: 0.6, w: 9.0, h: 0.6, fontFace: SANS, fontSize: 24, bold: true, color: ink });
  const c3 = (i, j) => ({ fontFace: SANS, fontSize: 14.5, bold: i === 0 || j === 0, color: i === 0 ? mute : j === 0 ? acc : body, border: [none, none, i === 0 ? bd("BBBBBB", 0.5) : none, none] });
  tbl(s, [["모델", "무엇을 보나"], ...MODELS], c3, { x: 3.6, y: 1.6, w: 9.0, colW: [0.9, 8.1], rowH: [0.42, 0.78, 0.78, 0.78] });
  T(s, "실내 온도에 따른 비중", { x: 3.6, y: 4.6, w: 4, h: 0.35, fontFace: SANS, fontSize: 13, bold: true, color: ink });
  tbl(s, [["", "A", "B", "C"], ...W], (i, j) => ({ fontFace: SANS, fontSize: 14, align: j ? "center" : "left", color: i === 0 ? mute : body, bold: i === 0, border: [none, none, i === 0 ? bd("BBBBBB", 0.5) : none, none] }),
    { x: 3.6, y: 5.0, w: 4.6, colW: [1.6, 1.0, 1.0, 1.0], rowH: 0.4 });
  T(s, NOTE5, { x: 8.7, y: 5.4, w: 3.9, h: 0.9, fontFace: SANS, fontSize: 13, color: body });
  T(s, "6", { x: 11.9, y: 6.95, w: 0.7, h: 0.3, fontFace: SANS, fontSize: 10, color: mute, align: "right" });
}

// ───────── 시안 4. 신문 지면형 ─────────
// 상단 굵은 이중선, 큰 명조 헤드라인, 본문을 세로 실선으로 나눈 단 구성, 장 꼬리표는 굵은 작은 글씨.
{
  const ink = "111111", body = "2F2F2F", mute = "777777", acc = "7A2E1E";
  let s = label(4, "신문 지면형");
  line(s, 0.7, 0.75, 11.93, ink, 3); line(s, 0.7, 0.85, 11.93, ink, 0.75);
  T(s, "제1회 전국 농과계 대학 농업 AI 경진대회", { x: 0.7, y: 0.95, w: 5.8, h: 0.35, fontFace: SANS, fontSize: 12, color: body });
  T(s, "2026년 11월 6일 · 대구 EXCO", { x: 6.6, y: 0.95, w: 6.03, h: 0.35, fontFace: SANS, fontSize: 12, color: body, align: "right" });
  T(s, TITLE, { x: 0.7, y: 2.0, w: 11.9, h: 2.0, fontFace: SERIF, fontSize: 42, bold: true, color: ink });
  line(s, 0.7, 4.45, 11.93, ink, 0.75);
  T(s, "정형", { x: 0.7, y: 4.65, w: 3, h: 0.3, fontFace: SANS, fontSize: 11, bold: true, color: acc });
  T(s, "시설 딸기 배지의 온도와 EC를 온실 센서 기록으로 추정", { x: 0.7, y: 4.98, w: 5.6, h: 0.7, fontFace: SANS, fontSize: 15, color: body });
  vline(s, 6.67, 4.65, 1.1, "AAAAAA", 0.5);
  T(s, "이미지", { x: 7.0, y: 4.65, w: 3, h: 0.3, fontFace: SANS, fontSize: 11, bold: true, color: acc });
  T(s, "딸기 사진이 실제인지, AI로 만들거나 고친 것인지 판별", { x: 7.0, y: 4.98, w: 5.6, h: 0.7, fontFace: SANS, fontSize: 15, color: body });
  line(s, 0.7, 6.3, 11.93, ink, 0.75);
  T(s, "팜모니  ·  " + MEMBERS, { x: 0.7, y: 6.42, w: 11.9, h: 0.35, fontFace: SANS, fontSize: 12, color: body });

  s = label(4, "신문 지면형");
  line(s, 0.7, 0.45, 11.93, ink, 2.25);
  T(s, "정형 1.4  온도 기본 예측", { x: 0.7, y: 0.55, w: 6, h: 0.3, fontFace: SANS, fontSize: 11, bold: true, color: acc });
  T(s, LEAD, { x: 0.7, y: 0.9, w: 11.9, h: 0.75, fontFace: SERIF, fontSize: 30, bold: true, color: ink });
  line(s, 0.7, 1.8, 11.93, "AAAAAA", 0.5);
  const c4 = (i, j, last) => ({ fontFace: SANS, fontSize: 14.5, bold: i === 0 || j === 0, color: i === 0 ? mute : body, border: [none, none, i === last ? none : bd("CCCCCC", 0.5), none] });
  tbl(s, [["모델", "무엇을 보나"], ...MODELS], c4, { x: 0.7, y: 2.0, w: 7.2, colW: [0.9, 6.3], rowH: [0.42, 0.85, 0.85, 0.85] });
  T(s, CAP4, { x: 0.7, y: 5.15, w: 7.2, h: 0.6, fontFace: SANS, fontSize: 11.5, color: mute, italic: true });
  vline(s, 8.25, 2.0, 4.3, "AAAAAA", 0.5);
  tbl(s, [["실내 온도", "A", "B", "C"], ...W], (i, j, l) => Object.assign(c4(i, j, l), { align: j ? "center" : "left", bold: i === 0 }), { x: 8.6, y: 2.0, w: 4.0, colW: [1.45, 0.85, 0.85, 0.85], rowH: 0.55 });
  T(s, CAP5, { x: 8.6, y: 4.3, w: 4.0, h: 0.35, fontFace: SANS, fontSize: 11.5, color: mute, italic: true });
  T(s, NOTE5, { x: 8.6, y: 4.8, w: 4.0, h: 0.95, fontFace: SANS, fontSize: 14, color: body });
  line(s, 0.7, 6.85, 11.93, ink, 0.75);
  T(s, FOOT, { x: 0.7, y: 6.95, w: 10.5, h: 0.3, fontFace: SANS, fontSize: 9, color: mute });
  T(s, "6", { x: 11.9, y: 6.95, w: 0.7, h: 0.3, fontFace: SANS, fontSize: 10, bold: true, color: ink, align: "right" });
}

// ───────── 시안 5. 현장 기록지형 ─────────
// 옅은 미색 종이 바탕, 짙은 녹색 한 가지, 표는 가는 실선 격자(야장 느낌), 제목 왼쪽에 장 번호 숫자만 크게.
{
  const paper = "F7F4EC", ink = "263226", body = "3B4A3B", mute = "8A8F80", acc = "3E5C3A", grid = "C9C6B8";
  let s = label(5, "현장 기록지형");
  s.background = { color: paper };
  T(s, "현장 발표", { x: 0.9, y: 1.0, w: 4, h: 0.35, fontFace: SANS, fontSize: 13, bold: true, color: acc });
  T(s, "제1회 전국 농과계 대학 농업 AI 경진대회", { x: 0.9, y: 1.35, w: 8, h: 0.35, fontFace: SANS, fontSize: 13, color: mute });
  T(s, TITLE, { x: 0.9, y: 2.2, w: 11.5, h: 1.8, fontFace: SERIF, fontSize: 38, bold: true, color: ink });
  tbl(s, [["팀", "팜모니", "날짜", "2026. 11. 6."], ["구성", MEMBERS, "장소", "대구 EXCO"]],
    (i, j) => ({ fontFace: SANS, fontSize: 13, color: j % 2 ? body : mute, bold: false, border: [bd(grid, 0.75), bd(grid, 0.75), bd(grid, 0.75), bd(grid, 0.75)] }),
    { x: 0.9, y: 5.0, w: 11.5, colW: [0.9, 6.4, 0.9, 3.3], rowH: 0.45 });

  s = label(5, "현장 기록지형");
  s.background = { color: paper };
  T(s, "1.4", { x: 0.7, y: 0.5, w: 1.3, h: 0.8, fontFace: SERIF, fontSize: 34, bold: true, color: acc });
  T(s, "온도 기본 예측", { x: 2.0, y: 0.52, w: 6, h: 0.3, fontFace: SANS, fontSize: 12, color: mute });
  T(s, LEAD, { x: 2.0, y: 0.82, w: 10.6, h: 0.55, fontFace: SERIF, fontSize: 24, bold: true, color: ink });
  const c5 = (i, j) => ({ fontFace: SANS, fontSize: 14.5, bold: i === 0, color: i === 0 ? acc : body, border: [bd(grid, 0.75), bd(grid, 0.75), bd(grid, 0.75), bd(grid, 0.75)] });
  tbl(s, [["모델", "무엇을 보나"], ...MODELS], c5, { x: 0.9, y: 1.95, w: 6.9, colW: [0.9, 6.0], rowH: [0.45, 0.8, 0.8, 0.8] });
  T(s, CAP4, { x: 0.9, y: 5.0, w: 6.9, h: 0.6, fontFace: SANS, fontSize: 12, color: mute });
  tbl(s, [["실내 온도", "A", "B", "C"], ...W], (i, j) => Object.assign(c5(i, j), { align: j ? "center" : "left", fontSize: 15.5 }), { x: 8.25, y: 1.95, w: 4.35, colW: [1.5, 0.95, 0.95, 0.95], rowH: 0.58 });
  T(s, CAP5, { x: 8.25, y: 4.4, w: 4.35, h: 0.35, fontFace: SANS, fontSize: 12, color: mute });
  T(s, NOTE5, { x: 8.25, y: 4.9, w: 4.35, h: 0.95, fontFace: SANS, fontSize: 14.5, color: body });
  T(s, FOOT, { x: 0.9, y: 6.95, w: 10.5, h: 0.3, fontFace: SANS, fontSize: 9, color: mute });
  T(s, "6", { x: 11.9, y: 6.95, w: 0.7, h: 0.3, fontFace: SANS, fontSize: 10, color: mute, align: "right" });
}

pres.writeFile({ fileName: path.join(__dirname, "템플릿시안_5종_v2.pptx") }).then((f) => console.log("ok", f));
