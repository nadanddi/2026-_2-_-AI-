// 팜모니 AI 발표자료 초안 v3 생성기 (v2 + 색: 연두·흰색·검정·회색) (2026-11-02, 집 클로드)
// 색 4가지만: 연두 ACC(면·막대·검정 바탕 위 글자에만), 흰색 BG, 회색 SUP(카드), 검정 INK(글자·어두운 면)
// 연두는 흰 바탕 위 글자로 쓰지 않음(대비 부족) — 흰·회색 바탕의 강조는 검정 굵은 글씨
// 헤더(섹션 이름)·제목·푸터(팀명·쪽번호) 위치는 모든 슬라이드에서 고정
// 실행: NODE_PATH=<pptxgenjs가 있는 node_modules> node build_deck_v1.js <출력.pptx>
const path = require("path");
const pptxgen = require("pptxgenjs");
const OUT = process.argv[2] || path.join(__dirname, "..", "팜모니AI_발표자료_초안v3.pptx");

const ACC = "92D050", BG = "FFFFFF", SUP = "EDEDED", INK = "111111";
const FONT = "맑은 고딕";
const THEME = {
  name: "Farmoni4", headFontFace: FONT, bodyFontFace: FONT,
  colors: { dk1: INK, lt1: BG, dk2: INK, lt2: SUP, accent1: ACC, accent2: INK, accent3: SUP, accent4: ACC, accent5: INK, accent6: SUP, hlink: ACC, folHlink: INK },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.author = "팜모니 AI"; pres.title = "팜모니 AI 발표자료";

const W = 13.333, M = 0.6, CW = W - 2 * M;
const HDR = { x: M + 0.3, y: 0.35, w: 9, h: 0.35 };
const TTL = { x: M, y: 0.72, w: CW, h: 0.8 };
const TOP = 1.8, BOT = 6.75;
const FOOT_TXT = "팜모니 AI  ·  제1회 전국 농과계 대학 농업 AI 경진대회";

function layout(name, bg, fg, hc) {
  pres.defineSlideMaster({
    title: name,
    background: { color: bg },
    objects: [
      { text: { text: FOOT_TXT, options: { x: M, y: 7.0, w: 8, h: 0.3, fontFace: FONT, fontSize: 10, color: fg, margin: 0 } } },
      { placeholder: { options: { name: "header", type: "body", x: HDR.x, y: HDR.y, w: HDR.w, h: HDR.h, fontFace: FONT, fontSize: 12, bold: true, color: hc, margin: 0, valign: "middle" }, text: "" } },
      { rect: { x: M, y: 0.435, w: 0.18, h: 0.18, fill: { color: ACC }, line: { color: ACC, width: 0 } } },
      { placeholder: { options: { name: "title", type: "title", x: TTL.x, y: TTL.y, w: TTL.w, h: TTL.h, fontFace: FONT, fontSize: 30, bold: true, color: fg, margin: 0, valign: "middle", align: "left" }, text: "" } },
    ],
    slideNumber: { x: W - M - 1, y: 7.0, w: 1, h: 0.3, fontFace: FONT, fontSize: 10, color: fg, align: "right", margin: 0 },
  });
}
layout("LIGHT", BG, INK, INK);
layout("DARK", INK, BG, ACC);

let n = 0;
function slide(kind, header, title, notes, section) {
  if (section) pres.addSection({ title: section });
  const s = pres.addSlide({ masterName: kind, sectionTitle: curSection });
  s.addText(header, { placeholder: "header" });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes.trim());
  n++;
  return s;
}
let curSection = "";
function section(t) { curSection = t; pres.addSection({ title: t }); }

// ---------- 작은 도우미 ----------
const tb = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, fontFace: FONT, color: INK, fontSize: 14, margin: 0, valign: "top" }, o));
const box = (s, x, y, w, h, fill, name) => s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: name || "box" });
const rbox = (s, x, y, w, h, fill, name) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: name || "card" });
function bullets(s, items, o) {
  const runs = items.map((t, i) => {
    const r = Array.isArray(t) ? t : [{ text: t }];
    return r.map((q, j) => ({ text: q.text, options: Object.assign({ bullet: j === 0 ? { indent: 14 } : undefined, breakLine: j === r.length - 1 && i < items.length - 1, paraSpaceAfter: 8 }, q.options || {}) }));
  }).flat();
  return tb(s, runs, o);
}
function stat(s, x, y, w, big, label, color, sz) {
  tb(s, big, { x, y, w, h: 0.95, fontSize: sz || 54, bold: true, color: color || INK, valign: "bottom" });
  tb(s, label, { x, y: y + 1.0, w, h: 0.5, fontSize: 14 });
}
function arrow(s, x, y, w, color) {
  s.addShape(pres.shapes.RIGHT_ARROW, { x, y, w, h: 0.42, fill: { color: color || INK }, line: { color: color || INK, width: 0 }, objectName: "arrow" });
}
const CH = { // 모든 차트의 글자 모양을 같게
  catAxisLabelColor: INK, valAxisLabelColor: INK, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 12,
  dataLabelFontFace: "+mn-lt", dataLabelFontSize: 12, dataLabelColor: INK, titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: INK,
  valGridLine: { color: SUP, size: 0.75 }, catGridLine: { style: "none" }, catAxisLabelRotate: 0, catAxisLineColor: INK, valAxisLineShow: false,
};
function table(s, rows, o) {
  const fs = o.fontSize || 14;
  const data = rows.map((r, i) => r.map(c => ({ text: String(c), options: { fontFace: FONT, fontSize: fs, color: i === 0 ? BG : INK, bold: i === 0, fill: { color: i === 0 ? INK : (i % 2 ? BG : SUP) }, valign: "middle", margin: 0.06 } })));
  s.addTable(data, Object.assign({ border: { type: "solid", pt: 0.5, color: SUP } }, o, { fontSize: undefined }));
}

// =====================================================================
section("공통");
{ // 1 표지
  const s = slide("DARK", "제1회 전국 농과계 대학 농업 AI 경진대회 · 현장 발표", "팜모니 AI", `
[말할 것] 팀 소개 한 줄.
[노트] 발표 내용은 최종 검증에 낸 답안 기준입니다. 정형 = submission_35e.csv(플랫폼 26회차, 24회차와 같은 답), 이미지 = 팜모니_나단11.zip(R21).`);
  tb(s, "딸기 온실 배지 온도·EC 예측  ×  생성·편집 사진 판별", { x: M, y: 2.3, w: CW, h: 0.6, fontSize: 24, bold: true, color: BG });
  tb(s, "기본 모델  →  자료에서 찾은 정보원  →  실행 전에 고정한 채택 기준", { x: M, y: 3.0, w: CW, h: 0.5, fontSize: 18, color: ACC, bold: true });
  box(s, M, 4.6, 0.08, 1.4, ACC, "mark");
  tb(s, [{ text: "발표 조나단", options: { bold: true, breakLine: true } }, { text: "팀원 조나단 · 박찬유 · 반지원 · 김태경 · 김우진", options: { breakLine: true } }, { text: "지도교수 양명균" }], { x: M + 0.3, y: 4.6, w: 9, h: 1.4, fontSize: 16, color: BG, paraSpaceAfter: 6 });
}
{ // 2 두 미션 한눈에
  const s = slide("LIGHT", "공통 · 한눈에", "두 미션의 최종 결과", `
[말할 것] 두 미션 모두 기본 모델을 세운 뒤, 자료를 뜯어보며 정보원을 찾아 붙였습니다. 붙일지는 실행 전에 정한 기준으로 판정했습니다.
[노트]
- 정형 최종: 24회차 submission_35e.csv, 공개 점수 온도 0.4608, EC 0.1148 (RMSE, 낮을수록 좋음). 26회차에 같은 답을 다시 제출했습니다(마감 직전 운영진이 1회 추가 허용). [정형 README ⑦]
- 이미지 최종: 나단11, 플랫폼 18회차 업로드(10-16 15:30), R21 채점, SHA256 6fe04909… [이미지 README 0·⑦]
- 두 점수 모두 공개 리더보드 값입니다. 검증 수치의 출처는 부록 A5에 있습니다.`);
  const cw = (CW - 0.5) / 2;
  [[M, "정형 미션", "두 온실의 시간별 배지 온도·EC 예측", [["0.4608", "온도 RMSE (℃)"], ["0.1148", "EC RMSE (dS/m)"]]],
   [M + cw + 0.5, "이미지 미션", "딸기 시설재배 사진의 진짜·가짜 판별", [["0.97", "Accuracy (R21)"], ["25.59", "환산점수"]]]].forEach(([x, h, d, st]) => {
    rbox(s, x, TOP, cw, 3.3, SUP);
    tb(s, h, { x: x + 0.35, y: TOP + 0.3, w: cw - 0.7, h: 0.45, fontSize: 20, bold: true });
    tb(s, d, { x: x + 0.35, y: TOP + 0.8, w: cw - 0.7, h: 0.4, fontSize: 14 });
    st.forEach(([b, l], i) => stat(s, x + 0.35 + i * (cw - 0.7) / 2, TOP + 1.25, (cw - 0.7) / 2, b, l));
  });
  rbox(s, M, 5.45, CW, 1.1, INK, "message");
  tb(s, "공통 방식 — 기본 모델을 세우고, 자료에서 정보원을 찾아 붙이고, 붙일지는 실행 전에 고정한 기준으로 판정", { x: M + 0.35, y: 5.45, w: CW - 0.7, h: 1.1, fontSize: 16, bold: true, color: BG, valign: "middle" });
  tb(s, "점수: 공개 리더보드(최종 제출 기준)", { x: M, y: 6.6, w: CW, h: 0.3, fontSize: 10 });
}

// =====================================================================
section("정형 미션");
{ // 3 문제와 자료에서 찾은 것
  const s = slide("LIGHT", "정형 미션 · 문제", "하루 기록이 섞여 있고, 같은 동이 ID를 오간다", `
[말할 것] 처음에는 시간별 회귀 문제로 봤습니다. 오차를 분석하다가, 하루 기록끼리 0시 센서값이 전날 23시와 이어진다는 것을 알았습니다. 그래서 "어제"를 찾을 수 있으면 정보가 늘어납니다.
[노트]
- 학습 자료: F13·F47 학습 기록 400일, 9,600행. [정형 README ④]
- 기록 번호 = 문제설명서의 "온실별 상대 일차". 179 미만이 1차 구간, 이상이 2차 구간이고, 평가 기록은 모두 2차 구간입니다. [정형 README 용어]
- 학습 정답으로 이어짐이 확실한 짝 가운데 약 4분의 1이 다른 온실 ID였습니다. 그래서 같은 동이 ID를 오간다고 판단했습니다. [정형 README ③]
- 이 구조는 대회 자료를 만드는 과정에서 생긴 성질이며 현장에서는 생기지 않습니다. 다룬 방법은 9장과 부록 A2.`);
  bullets(s, [
    [{ text: "과제  ", options: { bold: true } }, { text: "두 온실(F13·F47)의 시간별 배지 온도·EC 예측" }],
    [{ text: "평가  ", options: { bold: true } }, { text: "60기록 × 24시간, 모두 2차 구간" }],
    [{ text: "입력  ", options: { bold: true } }, { text: "그 시각까지의 실내·외기·구동기 기록" }],
    [{ text: "발견  ", options: { bold: true } }, { text: "0시 센서값이 어떤 기록의 23시와 이어짐 → '어제'를 찾으면 정보가 늘어남" }],
  ], { x: M, y: TOP, w: 5.4, h: 3.6, fontSize: 16 });
  // 오른쪽 그림: 두 ID 줄에 기록 상자, 같은 동의 이어짐을 빨간 화살표로
  const gx = 6.6, gw = 6.1;
  tb(s, "F13", { x: gx, y: TOP + 0.45, w: 0.6, h: 0.5, fontSize: 14, bold: true, valign: "middle" });
  tb(s, "F47", { x: gx, y: TOP + 2.05, w: 0.6, h: 0.5, fontSize: 14, bold: true, valign: "middle" });
  const bx = [gx + 0.8, gx + 2.15, gx + 3.5, gx + 4.85];
  const lab1 = ["학습", "평가", "학습", "평가"], lab2 = ["평가", "학습", "평가", "학습"];
  bx.forEach((x, i) => {
    rbox(s, x, TOP + 0.4, 1.1, 0.6, i % 2 ? INK : SUP); tb(s, lab1[i], { x, y: TOP + 0.4, w: 1.1, h: 0.6, fontSize: 14, align: "center", valign: "middle", color: i % 2 ? BG : INK });
    rbox(s, x, TOP + 2.0, 1.1, 0.6, i % 2 ? SUP : INK); tb(s, lab2[i], { x, y: TOP + 2.0, w: 1.1, h: 0.6, fontSize: 14, align: "center", valign: "middle", color: i % 2 ? INK : BG });
  });
  [[0, 0, 1, 1], [1, 1, 2, 0], [2, 0, 3, 0]].forEach(([a, ra, b, rb]) => {
    const x1 = bx[a] + 1.1, x2 = bx[b];
    let y1, y2;
    if (ra === rb) { y1 = y2 = (ra === 0 ? TOP + 0.7 : TOP + 2.3); }
    else { y1 = ra === 0 ? TOP + 1.0 : TOP + 2.0; y2 = rb === 0 ? TOP + 1.0 : TOP + 2.0; }
    s.addShape(pres.shapes.LINE, { x: x1, y: Math.min(y1, y2), w: x2 - x1, h: Math.max(Math.abs(y2 - y1), 0.001), flipV: y2 < y1, line: { color: ACC, width: 4, endArrowType: "triangle" }, objectName: "link" });
  });
  tb(s, "연두 화살표: 같은 동의 하루하루가 두 ID를 오감", { x: gx + 0.8, y: TOP + 2.85, w: gw - 0.8, h: 0.45, fontSize: 14, bold: true });
  rbox(s, gx, 5.3, gw, 1.2, ACC);
  stat(s, gx + 0.3, 5.05, 1.6, "1/4", "", INK, 40);
  tb(s, "학습 정답으로 이어짐이 확실한 짝 중 다른 ID인 비율", { x: gx + 2.0, y: 5.3, w: gw - 2.3, h: 1.2, fontSize: 14, valign: "middle" });
}
{ // 4 전략 3단계
  const s = slide("LIGHT", "정형 미션 · 전략", "세 단계: 기본 예측 → 어제 이어 붙이기 → 작은 보정", `
[말할 것] ①이 성능의 대부분을 냅니다. ②는 어제를 찾을 수 있을 때만 작동합니다. ③은 자료에서 잰 작은 보정입니다.
[노트]
- 처리 순서: 온도 NX → 어제 앵커 → TX3 → 예측 사슬 / EC pre → HR5 → 어제 앵커 → 학습 EC 범위로 자르기. [정형 README ⑤]
- 단계별로 평가 예측이 바뀐 범위(평가 60기록): 어제 앵커 23기록 552행, TX3 13기록 144행, 사슬 17기록 408행, EC HR5 60기록 전부, EC 앵커 23기록. [정형 README ⑤ 표]
- 어제 이어 붙이기는 60기록 중 40기록에 작용했고, 나머지 20기록은 기본 예측과 TX3만으로 정해집니다.`);
  const bw = 3.55, gap = (CW - 3 * bw) / 2;
  const st = [["1", "기본 예측", "그 시각까지의 입력만으로 온도·EC 예측\n물리식 + 트리 + TabPFN", "평가 60기록 전부", SUP, INK],
              ["2", "어제 이어 붙이기", "같은 동의 어제를 찾아\n어제 23시 값으로 오늘 앞부분 보정", "60기록 중 40기록", INK, BG],
              ["3", "작은 보정", "고EC 동 추운 시간 +0.25℃(TX3)\nEC 하루 수준 보정(HR5)", "TX3 13기록 · HR5 60기록", SUP, INK]];
  st.forEach(([k, h, d, f, fill, fg], i) => {
    const x = M + i * (bw + gap);
    rbox(s, x, TOP + 0.2, bw, 4.2, fill);
    tb(s, k, { x: x + 0.35, y: TOP + 0.4, w: 1, h: 0.9, fontSize: 48, bold: true, color: fill === INK ? ACC : INK });
    tb(s, h, { x: x + 0.35, y: TOP + 1.4, w: bw - 0.7, h: 0.5, fontSize: 20, bold: true, color: fg });
    tb(s, d, { x: x + 0.35, y: TOP + 2.0, w: bw - 0.7, h: 1.3, fontSize: 14, color: fg });
    tb(s, f, { x: x + 0.35, y: TOP + 3.55, w: bw - 0.7, h: 0.5, fontSize: 14, bold: true, color: fg });
    if (i < 2) arrow(s, x + bw + 0.05, TOP + 2.1, gap - 0.1, ACC);
  });
  tb(s, "기록 수: 최종 답안이 평가 예측을 바꾼 범위(평가 60기록 기준)", { x: M, y: 6.6, w: CW, h: 0.3, fontSize: 10 });
}
{ // 5 기본 예측
  const s = slide("LIGHT", "정형 미션 · ① 기본 예측", "추울 땐 TabPFN을 줄이고, EC는 하루 수준을 맞춘다", `
[말할 것] 온도는 세 모델을 섞되 추운 시간에는 TabPFN 비중을 줄였습니다. 학습 범위 밖의 추운 행에서 TabPFN이 크게 틀렸기 때문입니다. EC는 하루 안에서 수준이 통째로 어긋나는 오차가 커서, 하루 평균 수준을 따로 예측해 보정했습니다.
[노트]
- 온도 섞는 식: g = clip((실내온도 − 8)/2, 0, 1), pred = 0.4·BASE + (0.2 + 0.4(1−g))·CODEX + 0.4·g·TabPFN. [정형 README ②]
- 학습 행 가중 0.2: 입력이 복원·보간된 흔적이 있는 행과 잡음 낀 날(학습 행만 판정).
- 추운 행 근거: 학습형 섞기(STK1)는 추운 날 검증에서 크게 실패했습니다(≤ 8℃ 행 +67~+84%). 손으로 정한 "추우면 TabPFN 끄기"가 외삽 안전 장치였음을 확인했습니다. (검증, 카탈로그 7.014)
- EC 남은 오차의 65~77%가 고EC 기록 35개에 몰렸고, 하루 통째로 어긋나는 몫이 78~92%였습니다. (검증, 카탈로그 6.756) → HR5 설계 근거.
- TabPFN v2: 사전학습 가중치 그대로, 미세조정 없음. 학습 행 2,000개를 참고 예시로 넣어 예측.`);
  const cw = (CW - 0.5) / 2;
  [[M, "온도 (NX)", ["BASE: 물리 선형식 + LightGBM", "CODEX: Ridge 물리 기준선 + LightGBM", "TabPFN v2: 사전학습 표 모델(미세조정 없음)"], "실내온도 8~10℃ 아래에서 TabPFN 비중 0", "추운 날을 학습 범위 밖으로 둔 검증에서 TabPFN이 크게 틀림 → 손으로 정한 안전 장치"],
   [M + cw + 0.5, "EC", ["트리·신경망 0.8 + TabPFN 0.2", "하루 안 누적 평균과 반반 섞는 일내 수축", "HR5: 그 기록의 하루 평균 잔차를 3시간마다 예측"], "남은 오차의 78~92%가 '하루 통째' 어긋남", "그래서 시간별 모양보다 하루 수준을 따로 맞춤"]].forEach(([x, h, items, key, why]) => {
    tb(s, h, { x, y: TOP, w: cw, h: 0.45, fontSize: 20, bold: true });
    bullets(s, items, { x, y: TOP + 0.6, w: cw, h: 1.8, fontSize: 14 });
    rbox(s, x, TOP + 2.55, cw, 2.2, SUP);
    tb(s, key, { x: x + 0.3, y: TOP + 2.75, w: cw - 0.6, h: 0.8, fontSize: 18, bold: true });
    tb(s, why, { x: x + 0.3, y: TOP + 3.6, w: cw - 0.6, h: 1.0, fontSize: 14 });
  });
}
{ // 6 어제 이어 붙이기
  const s = slide("LIGHT", "정형 미션 · ② 어제 이어 붙이기", "어제 23시 값이 오늘 새벽 예측을 가장 크게 바꾼다", `
[말할 것] 배지 온도와 EC는 자정을 넘어 이어집니다. 어제 23시 값을 알면 오늘 새벽 예측이 크게 좋아지고, 보정 강도는 시간이 갈수록 줄게 했습니다.
[노트]
- 앵커 식: pred_h = q_h + β_h·clip(A − 0.36 − q_0, −2, 2). β_h는 1차 구간 학습 정답으로 잰 시각별 기울기입니다. 0시 .76 → 9시 .13, 그 뒤 .16~.24 (make_temp_v22_anchor.py 17행). [정형 README ②]
- 적용 조건: 링커 확신 p ≥ .8이고 실내 공기 자정 변화 |dT| ≤ 1.5℃. 또는 (TAG2) .5 ≤ p < .8이면서 외기만 본 판단이나 실내만 본 판단이 같은 어제를 고른 경우.
- 사슬 식: pred_h += 0.9·exp(−h/8)·β_h·clip(A − 0.36 − x_0, −2, 2), A = 앞 기록의 최종 23시 예측.
- 규정: 평가 기록 사이의 정보 전달은 사슬 하나뿐입니다. 주최 측 10/16 답변 2항(앞선 평가 기록에 고정 규칙이 낸 값을 뒤 기록의 고정식 입력으로 씀)에 해당한다고 판단했습니다. [정형 README ③]
- 링커 반사실 검사: 평가 기록을 후보에서 모두 빼도 정답 앵커 15기록의 어제는 그대로였습니다 → 앵커는 평가 기록의 존재와 무관합니다.
- 금지된 W2(평가 자료에 앞 기록이 있는지로 갈리는 규칙)는 최종 답안에 없습니다.`);
  // 왼쪽 흐름
  const lx = M, lw = 5.6;
  rbox(s, lx, TOP, lw, 0.9, INK);
  tb(s, "링커: 0시 입력의 이어짐으로 '같은 동의 어제'를 고름", { x: lx + 0.3, y: TOP, w: lw - 0.6, h: 0.9, fontSize: 14, bold: true, color: BG, valign: "middle" });
  s.addShape(pres.shapes.DOWN_ARROW, { x: lx + lw / 2 - 0.2, y: TOP + 1.0, w: 0.4, h: 0.4, fill: { color: ACC }, line: { color: ACC, width: 0 }, objectName: "arrow" });
  const hw = (lw - 0.3) / 2;
  [[lx, "어제에 실측 정답 있음", "앵커: 어제 23시 정답으로 오늘 앞부분 보정", "23기록"], [lx + hw + 0.3, "어제가 평가 기록", "사슬: 어제의 최종 예측을 이어 씀", "17기록"]].forEach(([x, h, d, k]) => {
    rbox(s, x, TOP + 1.5, hw, 3.0, SUP);
    tb(s, h, { x: x + 0.25, y: TOP + 1.65, w: hw - 0.5, h: 0.5, fontSize: 14, bold: true });
    tb(s, d, { x: x + 0.25, y: TOP + 2.2, w: hw - 0.5, h: 1.0, fontSize: 14 });
    tb(s, k, { x: x + 0.25, y: TOP + 3.35, w: hw - 0.5, h: 0.9, fontSize: 36, bold: true, valign: "bottom" });
  });
  tb(s, "평가 60기록 기준 · 링커 확신이 낮거나 자정 기온 변화가 크면 쓰지 않음", { x: lx, y: TOP + 4.6, w: lw, h: 0.35, fontSize: 10 });
  // 오른쪽 β_h 차트
  const beta = [0.7648, 0.6232, 0.5525, 0.4844, 0.4091, 0.3453, 0.3025, 0.2576, 0.1866, 0.1264, 0.1570, 0.1661, 0.1655, 0.1566, 0.1999, 0.2032, 0.2233, 0.2263, 0.2145, 0.2008, 0.1897, 0.1721, 0.1991, 0.2355];
  s.addChart(pres.charts.BAR, [{ name: "β_h", labels: beta.map((_, h) => String(h)), values: beta }], Object.assign({}, CH, {
    x: 6.6, y: TOP, w: 6.1, h: 4.6, barDir: "col", chartColors: [ACC], showLegend: false, showValue: false, barGapWidthPct: 40,
    showTitle: true, title: "앵커 보정 강도 β_h (시각별, 학습 정답으로 잼)", valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.2,
    showCatAxisTitle: true, catAxisTitle: "시각(h)", catAxisTitleColor: INK, catAxisTitleFontSize: 12, catAxisTitleFontFace: "+mn-lt", objectName: "beta chart",
  }));
}
{ // 7 검증 방법
  const s = slide("LIGHT", "정형 미션 · 검증", "평가 구조를 흉내 낸 검증기와 실행 전 고정 기준", `
[말할 것] 평가 기록이 모두 2차 구간이고 추운 날이 섞여 있어서, 그 구조를 흉내 낸 검증기를 따로 만들었습니다. 기준을 실행 전에 정해 코드로 커밋한 뒤 실행했습니다.
[노트]
- 로컬 기준을 통과해 들어간 부품: TX3(44칸 모두 개선, P .0047), 사슬(P .0003), 사슬 확신 문턱 p ≥ .5(P .0029). [정형 README ⑤ 표]
- 기준을 다 채우지 못해 '제출 비교'로 채택한 부품도 있습니다: NX 섞기 비중, 계절 지표, 앵커, TAG2, CHT, HR5, EC 앵커. 이 사실을 그대로 밝힙니다(부록 A8).
- 검증 세계 주의: 판정 기록의 입력이 학습 특징의 이력에 섞이면 추운 날 성적이 낙관적이 됩니다. 판정은 같은 실행에서 재학습한 것끼리만 비교했습니다. (카탈로그 7.034)
- 검증기는 같은 학습 자료 안에서 기록을 나눈 것이고, 일부는 값 선택에도 썼습니다(부록 A5).
- 규정 검사: 미래 입력 절단 3곳(F13 206일 4시, F13 237일 3시, F47 202일 5시)과 다른 온실 격리 2개에서 예측이 비트 단위로 같았습니다. [정형 README ③]`);
  table(s, [["검증기", "가리는 것", "노린 상황"],
            ["DIAG10", "학습 기록 10겹 무작위", "전체 평균 성능"],
            ["EL1 · P2LOO", "2차 구간 기록 묶음 · 한 기록씩", "평가와 같은 2차 구간"],
            ["EXT10 · EXT12", "하루 최저 10·12℃ 미만인 날", "학습 범위 밖의 추운 날"]],
        { x: M, y: TOP, w: 7.2, colW: [2.0, 2.9, 2.3], rowH: 0.62, fontSize: 14 });
  tb(s, "검증기는 모두 F13·F47 학습 기록을 기록 단위로 가린 것", { x: M, y: TOP + 2.65, w: 7.2, h: 0.35, fontSize: 10 });
  const rx = 8.3, rw = W - M - rx;
  rbox(s, rx, TOP, rw, 2.5, INK);
  tb(s, [{ text: "채택 기준", options: { bold: true, color: ACC, breakLine: true } }, { text: "실행 전에 코드로 고정", options: { breakLine: true } }, { text: "모든 검증기·시드에서 같은 방향 + 블록 부트스트랩 P(악화) < .025" }], { x: rx + 0.3, y: TOP + 0.2, w: rw - 0.6, h: 2.1, fontSize: 16, color: BG, paraSpaceAfter: 6 });
  rbox(s, rx, TOP + 2.75, rw, 2.0, SUP);
  tb(s, [{ text: "규정 검사 ALL PASS", options: { bold: true, breakLine: true } }, { text: "미래 입력 절단 3곳 · 다른 온실 격리 2개에서 그 이전 예측이 비트 단위로 같음" }], { x: rx + 0.3, y: TOP + 2.95, w: rw - 0.6, h: 1.6, fontSize: 14, paraSpaceAfter: 6 });
  rbox(s, M, TOP + 3.25, 7.2, 1.5, SUP);
  tb(s, [{ text: "로컬 기준을 통과한 부품  ", options: { bold: true } }, { text: "TX3 (P .0047) · 예측 사슬 (P .0003) · 사슬 문턱 (P .0029)", options: { breakLine: true } }, { text: "나머지 부품은 제출 비교로 채택 — 부록 A8에 부품별로 밝힘" }], { x: M + 0.3, y: TOP + 3.35, w: 6.6, h: 1.3, fontSize: 14, valign: "middle", paraSpaceAfter: 6 });
}
{ // 8 개선 과정과 실패
  const s = slide("LIGHT", "정형 미션 · 개선 과정", "26번의 제출: 기준을 통과한 변경만 리더보드에서도 좋아졌다", `
[말할 것] 사전 기준을 엄격히 통과한 변경은 리더보드에서도 좋아졌습니다. 기준을 근소하게 못 넘었거나 추운 영역을 건드린 변경은 크게 어긋났습니다. 금지 확인을 받은 W2 규칙은 확인 전에 스스로 빼고 그 비용을 감수했습니다.
[노트]
- 실패 사례: 10회차 EC 고EC 날 제외 학습 .3538 / 20회차 BASE 이력 재연결 .4754 / 22회차 범위 밖 보정·학습형 TX3 .5000 / 25회차 실험 .4845(되돌림). [제출이력_정형_2026-10-20.md]
- 성공 사례: 19회차(사슬, 로컬 P .0003) .4706, 21회차(사슬 문턱, P .0029) .4685 — 사전 기준을 통과해 리더보드로 옮겨진 예.
- W2: 11~15회차에 있었고, 현장형 정리를 하며 16회차(10-15)에 뺐습니다. EC가 .0979 → .1225로 나빠졌습니다. 주최 측이 10-16에 금지를 확인했고, 그 뒤 모든 제출에 없습니다.
- 리더보드는 후보 비교에만 썼습니다. 점수로 값을 계산하지 않았습니다. 예외 1건: TabPFN 비중 0.4(두 후보 중 선택, 부록 A1).
- 그래프: 팀 회차 1~25의 공개 점수. 26회차는 24회차와 같은 답의 재제출입니다.`);
  const temp = [.7450, .6666, .5624, .5697, .5456, .5251, .5193, .5160, .5127, .5126, .5089, .5012, .4883, .4864, .4853, .4861, .4817, .4922, .4706, .4754, .4685, .5000, .4665, .4608, .4845];
  const ec = [.2442, .2287, .2055, .2055, .2134, .2055, .2774, .1455, .1384, .3538, .1305, .1271, .1279, .1077, .0979, .1225, .1170, .1173, .1153, .1153, .1153, .1153, .1153, .1148, .1148];
  const lbl = temp.map((_, i) => String(i + 1));
  const cw = (CW - 0.5) / 2;
  [[M, "온도 RMSE (℃)", temp, 0.4, 0.8], [M + cw + 0.5, "EC RMSE", ec, 0, 0.4]].forEach(([x, t, v, lo, hi]) => {
    s.addChart(pres.charts.LINE, [{ name: t, labels: lbl, values: v }], Object.assign({}, CH, {
      x, y: TOP, w: cw, h: 3.4, chartColors: [INK], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 6, showLegend: false,
      showTitle: true, title: t + " · 팀 회차별 공개 점수", valAxisMinVal: lo, valAxisMaxVal: hi, catAxisLabelFontSize: 10, objectName: t,
    }));
  });
  const cards = [["실패", "10회차 EC .3538 · 22회차 온도 .5000\n기준 근소 미달·추운 영역을 건드린 변경"], ["통과 → 개선", "19회차 사슬(P .0003) .4706\n21회차 사슬 문턱(P .0029) .4685"], ["스스로 뺀 규칙", "W2를 금지 확인 전(10-15)에 제거\nEC .0979 → .1225를 감수"]];
  const kw = (CW - 0.6) / 3;
  cards.forEach(([h, d], i) => {
    const x = M + i * (kw + 0.3);
    rbox(s, x, TOP + 3.6, kw, 1.35, i === 2 ? INK : SUP);
    tb(s, h, { x: x + 0.25, y: TOP + 3.7, w: kw - 0.5, h: 0.4, fontSize: 14, bold: true, color: i === 2 ? ACC : INK });
    tb(s, d, { x: x + 0.25, y: TOP + 4.1, w: kw - 0.5, h: 0.8, fontSize: 14, color: i === 2 ? BG : INK });
  });
}
{ // 9 농업적 해석과 현장 적용
  const s = slide("LIGHT", "정형 미션 · 해석과 현장 적용", "어제 이어 붙이기를 끄면 오차가 3~31% 늘어난다", `
[말할 것] 현장에서는 어제가 달력으로 정해지므로 링커가 필요 없습니다. 어제의 내 예측을 이어 쓰는 사슬은 현장에서도 됩니다. 어제 실측 정답을 쓰는 앵커는 전날 배지 센서가 있어야 합니다. 그 차이를 숫자로 보였습니다.
[노트]
- 그래프: ABL1 끄기 실험(11-02 추가 실시, 서술용, 실행 전 고정 커밋 45ac69c5). 학습 자료 검증기의 2차 행 RMSE에서 '일반 방법만(온도 NX+TX3, EC pre+HR5)'이 최종보다 얼마나 나쁜가.
  온도: EXT10 .8387 → .7599(+10.4%), EXT12 .9059 → .8791(+3.1%), DIAG10 .7388 → .7146(+3.4%)
  EC: DIAG10 .2522 → .2329(+8.3%), EL1 .2787 → .2135(+30.5%), P2LOO .1913 → .1723(+11.0%)
- 온도에서 사슬 몫(현장에서도 가능)은 0.4~3.1%p, 앵커 몫은 EXT10 8%p·DIAG10 3%p입니다.
- 해석 ① 근거: 앵커 강도 β_h를 학습 정답으로 쟀고, 새벽에 크고 낮 이후 2할 정도 남습니다(6장). 배지의 열·수분 특성에 대한 일반론은 팀의 해석입니다.
- 해석 ② 근거: TX3 0.25℃는 학습 정답으로 잰 고EC 동의 '배지온도 − 입력 기반 예측' 차입니다(로컬 기준 통과, P .0047). 원인(배지 조성·관수 등)은 미확인입니다.
- 대회 자료 성질에 기대는 부분: 링커, excl 특징, 계절 지표(외기 쌍둥이로 2차 날짜 복원), 사슬의 연결. 현장 대응은 부록 A2.
- 미실시: 계절 지표까지 뺀 성능, 실제 어제로 사슬을 거는 현장형 사슬 성능.
- 한계: 검증기는 같은 학습 자료 안에서 나눈 것이고, 2차 판정 기록 수가 적습니다(EXT10 9기록, EXT12 26기록, DIAG10 52기록).`);
  s.addChart(pres.charts.BAR, [{ name: "끄면 오차 증가(%)", labels: ["온도 EXT10", "온도 EXT12", "온도 DIAG10", "EC DIAG10", "EC EL1", "EC P2LOO"], values: [10.4, 3.1, 3.4, 8.3, 30.5, 11.0] }], Object.assign({}, CH, {
    x: M, y: TOP, w: 6.6, h: 4.5, barDir: "bar", chartColors: [ACC], showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"",
    showTitle: true, title: "끄면 늘어나는 오차 (검증기 2차 행 RMSE, %)", valAxisMinVal: 0, valAxisMaxVal: 35, valAxisHidden: true, catAxisOrientation: "maxMin", objectName: "ablation chart",
  }));
  tb(s, "ABL1 끄기 실험(11-02, 서술용) · 학습 기록을 가린 검증기 · 평가 자료 아님", { x: M, y: TOP + 4.55, w: 6.6, h: 0.35, fontSize: 10 });
  const rx = 7.6, rw = W - M - rx;
  const cards = [["해석 ①", "배지 상태는 밤을 넘어 이어진다 — 전날 마지막 값이 새벽 예측의 가장 큰 정보", SUP, INK],
                 ["해석 ②", "고EC 동은 추운 시간에 배지 온도가 입력 기반 예측보다 높다(+0.25℃까지)", SUP, INK],
                 ["현장에서", "링커 불필요(달력) · 사슬은 그대로 가능 · 앵커는 전날 배지 실측이 있을 때만", INK, BG]];
  cards.forEach(([h, d, fill, fg], i) => {
    const y = TOP + i * 1.6;
    rbox(s, rx, y, rw, 1.4, fill);
    tb(s, h, { x: rx + 0.25, y: y + 0.12, w: rw - 0.5, h: 0.35, fontSize: 14, bold: true, color: fill === INK ? ACC : INK });
    tb(s, d, { x: rx + 0.25, y: y + 0.5, w: rw - 0.5, h: 0.85, fontSize: 14, color: fg });
  });
}

// =====================================================================
section("이미지 미션");
{ // 10 문제와 전략
  const s = slide("LIGHT", "이미지 미션 · 전략", "기본 판별 위에 원본 대조와 유형별 전문가를 얹는다", `
[말할 것] 사전학습 CLIP 특징과 잡음·JPEG 같은 저수준 통계를 반반 섞어 기본 판별을 만들었습니다. 기본 판별이 놓치는 유형은 작은 전문가들로 따로 잡았습니다.
[노트]
- 확률 계산 순서 1~7과 문턱은 [이미지 README ⑤ 표](부록 A4). 판정 임계값 0.5, 별도 확률 보정 없음.
- 전문가는 하나라도 자기 문턱을 넘으면 가짜 쪽으로 올립니다. 문턱은 실제 사진을 0.25~0.4%만 뒤집는 값으로 학습 자료에서 정했습니다.
- 모든 계산은 이미지 한 장 단위로 독립입니다. 순서 독립 검사 OK. [이미지 README ③]`);
  const bw = 3.55, gap = (CW - 3 * bw) / 2;
  const st = [["기본 판별", ["CLIP ViT-L/14 의미 특징 + 로지스틱", "저수준 통계 38개(잡음·선명도·JPEG 격자) + 부스팅", "두 확률의 평균"], SUP, INK],
              ["원본 대조", ["AI Hub 사진 152,043장의 지문 DB", "원본이 있는데 국소적으로 다르면 → 가짜 쪽", "원본과 거의 같으면 → 진짜 쪽"], INK, BG],
              ["유형별 전문가", ["부분 생성 · 부분 스타일 · 국소 조각", "필터 · GPT 재그리기 · GPT 편집", "하나라도 문턱을 넘으면 가짜 쪽"], SUP, INK]];
  st.forEach(([h, items, fill, fg], i) => {
    const x = M + i * (bw + gap);
    rbox(s, x, TOP + 0.1, bw, 4.4, fill);
    tb(s, String(i + 1), { x: x + 0.35, y: TOP + 0.3, w: 1, h: 0.9, fontSize: 48, bold: true, color: fill === INK ? ACC : INK });
    tb(s, h, { x: x + 0.35, y: TOP + 1.3, w: bw - 0.7, h: 0.5, fontSize: 20, bold: true, color: fg });
    bullets(s, items, { x: x + 0.35, y: TOP + 1.95, w: bw - 0.7, h: 2.4, fontSize: 14, color: fg });
    if (i < 2) arrow(s, x + bw + 0.05, TOP + 2.1, gap - 0.1, ACC);
  });
  tb(s, "모든 계산은 이미지 한 장 단위로 독립 · 판정 임계값 0.5", { x: M, y: 6.6, w: CW, h: 0.3, fontSize: 10 });
}
{ // 11 데이터 확보·생성
  const s = slide("LIGHT", "이미지 미션 · 데이터", "같은 사진으로 진짜·가짜를 만들어 편집 흔적을 배우게 했다", `
[말할 것] 공개 샘플이 21장뿐이라 학습 자료를 직접 만들었습니다. 공개 샘플과 촬영 환경이 가장 비슷한 공개 데이터셋을 찾아, 같은 사진으로 진짜와 가짜를 함께 만들었습니다. 모델이 사진 내용이 아니라 편집 흔적을 보게 하려는 것입니다.
[노트]
- 학습 이미지 17,160장(목록 17,164장 중 4장은 코드 규칙으로 제외). 기본 판별기는 실제 2,467 + 가짜 7,124. [이미지 README ①·④]
- AI Hub 원천 tar 306개를 이용 신청·승인 후 받았습니다. 촬영 묶음·품종·농장이 고르게 되도록 뽑았고, 대회 이미지에 맞춰 고르지 않았습니다. [이미지 data/README_data.md 1절]
- ChatGPT에 넣은 AI Hub 원본은 서로 다른 525장이고, 만든 이미지는 600장입니다(25장은 원본 없이 문장만으로). 부록 A6.
- 처리 흔들기: 실제 사진을 축소 방식·JPEG 품질을 바꿔 다시 저장해, 약한 처리 차이를 가짜로 보지 않게 했습니다.
- 재배포 제한 때문에 AI Hub 소재 16,197장은 파일 대신 원본 대응 해시로, 문장만으로 만든 967장은 파일로 최종 검증 자료에 냈습니다.`);
  const sw = (CW - 0.9) / 4;
  [["17,160", "최종 학습 이미지(장)"], ["6", "문제설명서 가짜 유형 모두 직접 제작"], ["600", "ChatGPT 생성·편집(장)"], ["152,043", "지문 DB 원천 사진(장)"]].forEach(([b, l], i) => {
    const x = M + i * (sw + 0.3);
    rbox(s, x, TOP, sw, 2.0, i === 0 ? INK : SUP);
    tb(s, b, { x: x + 0.25, y: TOP + 0.2, w: sw - 0.5, h: 0.95, fontSize: 40, bold: true, color: i === 0 ? ACC : INK, valign: "bottom" });
    tb(s, l, { x: x + 0.25, y: TOP + 1.25, w: sw - 0.5, h: 0.6, fontSize: 14, color: i === 0 ? BG : INK });
  });
  const cw = (CW - 0.5) / 2;
  tb(s, "실제 사진", { x: M, y: TOP + 2.35, w: cw, h: 0.45, fontSize: 18, bold: true });
  bullets(s, ["AI Hub 「지능형 수직농장 통합 데이터(딸기)」(596), 이용 신청·승인", "촬영 묶음·품종·농장이 고르게 뽑음", "대회 배포 형식으로 변환: 768×1024, JPEG Q92, 4:2:0"], { x: M, y: TOP + 2.9, w: cw, h: 2.0, fontSize: 14 });
  tb(s, "가짜 사진", { x: M + cw + 0.5, y: TOP + 2.35, w: cw, h: 0.45, fontSize: 18, bold: true });
  bullets(s, ["필터·합성·부분 생성·부분 스타일·전체 생성·전체 스타일", "SD1.5 · SDXL · Qwen-Image-Edit · 프로그램 처리", "ChatGPT 편집 — AI Hub 원본 525장 입력"], { x: M + cw + 0.5, y: TOP + 2.9, w: cw, h: 2.0, fontSize: 14 });
}
{ // 12 검증과 개선 과정
  const s = slide("LIGHT", "이미지 미션 · 검증과 개선", "실제 사진 오판이 늘지 않는 변경만 남겼다", `
[말할 것] 같은 원본의 진짜와 가짜가 학습과 검증에 갈라지지 않게 촬영 묶음 단위로 나눴습니다. 후보는 봉인 묶음에서 실제 사진 오판이 늘지 않는지를 먼저 봤습니다.
[노트]
- 회차 흐름: 나단1 .90(CLIP + 저수준 + 원본 대조) → 나단3 시간 초과로 반려(실패) → 나단6 .95(전문가 + 정밀 원본 대조) → 나단9 .97(+ GPT 재그리기 전문가) → 나단11 .97, 환산 25.59(+ GPT 편집 전문가). 팀원 기존 방식은 .77. [성능_정리 6절]
- 실패와 수정: E_gpt 첫 판은 GPT 가짜 26장 중 14장을 새로 잡았지만, 실제 오탐 기준을 근소하게 넘어 불통과였습니다. 원인이 '진짜 확정' 사진임을 찾아 그 사진은 제외했습니다. 불통과를 본 뒤 같은 자료로 정한 변형이라 낙관적일 수 있습니다. [이미지 README ⑤]
- 10-16에 내부 불통과로 버린 후보: 필터 전문가 재학습(C1, R3 하락), 토큰 LR(A2, R3~R5 하락). [성능_정리 6절]
- 최종 선택: 사전 규칙은 'LB ≥ .98이면 유지'였습니다. Accuracy 표시는 둘 다 .97이었지만 환산점수(23.82 → 25.59)와 내부 검증을 보고 나단11을 유지했습니다. [이미지 README ⑤]
- 봉인 R2~R5는 후보 채택 판정에 썼으므로 선택에 쓰인 자료이며, 출처는 학습과 같은 AI Hub 데이터셋입니다(부록 A5).`);
  s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: ["팀원 기존", "나단1", "나단6", "나단9", "나단11(최종)"], values: [0.77, 0.90, 0.95, 0.97, 0.97] }], Object.assign({}, CH, {
    x: M, y: TOP, w: 6.6, h: 4.5, barDir: "col", chartColors: [INK, INK, INK, INK, ACC], showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
    showTitle: true, title: "공개 리더보드 Accuracy (제출 판별)", valAxisMinVal: 0.6, valAxisMaxVal: 1.0, valAxisHidden: true, objectName: "image score chart",
  }));
  tb(s, "나단3은 추론 시간 초과로 반려(채점 없음) · 공개 리더보드 값", { x: M, y: TOP + 4.55, w: 6.6, h: 0.35, fontSize: 10 });
  const rx = 7.6, rw = W - M - rx;
  [["봉인 검증 묶음 R2~R5", "학습과 촬영 묶음이 겹치지 않게 떼어 둠 · 후보 판정 때만 열기"], ["촬영 묶음 단위 교차검증", "같은 원본의 진짜·가짜가 다른 폴드로 갈리지 않음"], ["문턱 = 실제 오탐률", "전문가 문턱은 실제 사진을 0.25~0.4%만 뒤집는 값"]].forEach(([h, d], i) => {
    const y = TOP + i * 1.6;
    rbox(s, rx, y, rw, 1.4, i === 0 ? INK : SUP);
    tb(s, h, { x: rx + 0.25, y: y + 0.12, w: rw - 0.5, h: 0.4, fontSize: 14, bold: true, color: i === 0 ? ACC : INK });
    tb(s, d, { x: rx + 0.25, y: y + 0.55, w: rw - 0.5, h: 0.8, fontSize: 14, color: i === 0 ? BG : INK });
  });
}
{ // 13 현장 적용 조건·한계
  const s = slide("LIGHT", "이미지 미션 · 현장 적용", "원본 DB가 없는 현장에서는 실제 사진 오탐이 늘어난다", `
[말할 것] 원본 대조는 AI Hub에 원본이 있는 사진에서만 작동합니다. 현장 사진에는 원본이 없으므로 이 부분이 빠진 성능으로 봐야 합니다. 빠지면 가짜 탐지는 거의 그대로이고 실제 사진 오탐이 늘어납니다.
[노트]
- 끄기 실험(10-20, 계획 사전 고정, 서술용): 균형 정확도 R2 .964 → .945, R3 .962 → .948, R4 .947 → .924, R5 .944 → .921. 공개 21장은 21/21 그대로. [이미지 README ⑤]
- 실제 오탐: R5 1.4% → 5.1%, 처리 변형 실제(webp) 37.5% → 75.8%. 가짜 탐지 기여는 작음(부분 생성 82.3 → 79.2%).
- R1(색 지문)만 끄면 영향이 작고(−0.4~+0.1%p), 차이는 대부분 정밀 대조의 '진짜 확정'에서 옵니다.
- 형식 게이트: JPEG·4:2:0·Q92 양자화표일 때만 전문가 3개(H·E_gr·E_gpt)를 켭니다. 현장 사진 형식이 다르면 이 셋이 빠집니다. 이 게이트를 끈 성능은 미실시입니다(부록 A4).
- 학습 자료와 출처가 다른 자료로 확인한 성능: 미실시(봉인 묶음도 같은 데이터셋).
- 지문 DB는 사진별 아주 작은 축소본(12×16 컬러, 24×32 흑백)을 담고 있습니다. 이용조건 판단은 부록 A6.`);
  s.addChart(pres.charts.BAR, [
    { name: "지문 DB 켬(최종)", labels: ["R2", "R3", "R4", "R5"], values: [96.4, 96.2, 94.7, 94.4] },
    { name: "지문 DB 끔", labels: ["R2", "R3", "R4", "R5"], values: [94.5, 94.8, 92.4, 92.1] }], Object.assign({}, CH, {
    x: M, y: TOP, w: 6.6, h: 4.5, barDir: "col", chartColors: [INK, ACC], showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: INK,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", showTitle: true, title: "봉인 묶음 균형 정확도 (%)", valAxisMinVal: 85, valAxisMaxVal: 100, valAxisHidden: true, objectName: "fingerprint ablation chart",
  }));
  tb(s, "지문 DB 끄기 실험(10-20, 서술용) · 봉인 검증 묶음 R2~R5", { x: M, y: TOP + 4.55, w: 6.6, h: 0.35, fontSize: 10 });
  const rx = 7.6, rw = W - M - rx;
  stat(s, rx, TOP - 0.1, rw, "1.4% → 5.1%", "지문 DB를 끄면 R5 실제 사진 오탐", INK, 40);
  rbox(s, rx, TOP + 1.65, rw, 1.35, SUP);
  tb(s, [{ text: "원본 대조가 작동하는 사진", options: { bold: true, breakLine: true } }, { text: "AI Hub DB에 원본이 있는 사진만 — 현장 사진은 대부분 해당 없음" }], { x: rx + 0.25, y: TOP + 1.75, w: rw - 0.5, h: 1.15, fontSize: 14, paraSpaceAfter: 4 });
  rbox(s, rx, TOP + 3.15, rw, 1.35, INK);
  tb(s, [{ text: "형식 게이트", options: { bold: true, color: ACC, breakLine: true } }, { text: "JPEG Q92 4:2:0이 아니면 전문가 3개가 꺼짐 — 끈 성능은 미실시", options: { color: BG } }], { x: rx + 0.25, y: TOP + 3.25, w: rw - 0.5, h: 1.15, fontSize: 14, paraSpaceAfter: 4 });
}

// =====================================================================
section("마무리");
{ // 14 AI 도구와 팀의 판단
  const s = slide("LIGHT", "공통 · AI 도구 활용", "무엇을 채택하고 어디까지 쓸지는 팀이 정했다", `
[말할 것] AI 도구로 실험 속도를 높였습니다. 무엇을 채택하고 어디까지 쓸지는 팀이 실행 전에 기준으로 정했습니다.
[노트]
- 예측(추론)에는 AI 도우미를 쓰지 않습니다. [정형 README ①, 이미지 README ①]
- 팀 판단의 예: '예측 시점 이전 정보까지만' 경계(10-13), W2를 주최 측 확인 전에 뺀 것(10-15), 실행 전 기준 고정 원칙.
- 사용 도구: Anthropic Claude, OpenAI Codex(코드·실험), ChatGPT 내장 이미지 생성·편집(학습용 가짜 600장).`);
  const cw = (CW - 0.5) / 2;
  [[M, "팀이 판단·설계", ["문제 정의", "채택 기준 — 실행 전 고정, 모든 검증기 같은 방향", "규정 경계 — '예측 시점 이전 정보까지만'", "W2 제거와 현장형 정리", "제출 후보와 최종본 선택", "AI Hub 데이터셋 선택·이용 신청"], SUP, INK],
   [M + cw + 0.5, "AI 도구(Claude · Codex)가 수행", ["코드 작성", "실험 실행", "가설·실험 제안", "GPT 이미지 생성 작업", "문서 초안"], INK, BG]].forEach(([x, h, items, fill, fg]) => {
    rbox(s, x, TOP, cw, 4.8, fill);
    tb(s, h, { x: x + 0.35, y: TOP + 0.3, w: cw - 0.7, h: 0.5, fontSize: 20, bold: true, color: fill === INK ? ACC : INK });
    bullets(s, items, { x: x + 0.35, y: TOP + 1.0, w: cw - 0.7, h: 3.6, fontSize: 16, color: fg });
  });
}
{ // 15 정리와 향후 계획
  const s = slide("LIGHT", "공통 · 정리", "한 것과 아직 하지 않은 것", `
[말할 것] 한 것과 향후 계획을 구분해 말합니다. 향후 항목은 모두 "아직 하지 않았다"고 분명히 말합니다.
[노트]
- 재현: 정형은 저장 중간값으로 바이트 단위 재현됩니다. 전체 재학습 시 EC 최대 .013 차이가 나며 원인은 TabPFN을 GPU 대신 CPU로 돌린 것입니다. 이미지는 제출 ZIP을 CPU로 재추론해 공개 21/21입니다. [정형 README ⑥, 이미지 README ⑥]`);
  const lw = 7.2;
  tb(s, "한 것", { x: M, y: TOP, w: lw, h: 0.45, fontSize: 20, bold: true });
  [["정형", "기본 예측 + 어제 이어 붙이기 — 규칙성에 기대는 몫을 숫자로 분리(+3~31%)"], ["이미지", "기본 판별 + 원본 대조 + 유형별 전문가 — 원본 대조 몫을 숫자로 분리(−1.4~−2.3%p)"], ["공통", "실행 전 기준 고정 · 규정 검사 · 바이트 단위 재현"]].forEach(([k, d], i) => {
    const y = TOP + 0.65 + i * 1.35;
    tb(s, k, { x: M, y, w: 1.3, h: 1.1, fontSize: 18, bold: true, valign: "middle" });
    rbox(s, M + 1.4, y, lw - 1.4, 1.1, SUP);
    tb(s, d, { x: M + 1.65, y, w: lw - 1.9, h: 1.1, fontSize: 14, valign: "middle" });
  });
  const rx = M + lw + 0.5, rw = W - M - rx;
  rbox(s, rx, TOP, rw, 4.7, INK);
  tb(s, "아직 하지 않은 것", { x: rx + 0.3, y: TOP + 0.25, w: rw - 0.6, h: 0.45, fontSize: 20, bold: true, color: ACC });
  bullets(s, ["실제 어제로 사슬을 거는 현장형 사슬 검증", "계절 지표를 달력 날짜로 바꾼 검증", "다른 출처 사진으로 이미지 판별 검증", "형식 게이트를 끈 이미지 성능"], { x: rx + 0.3, y: TOP + 0.9, w: rw - 0.6, h: 3.6, fontSize: 16, color: BG });
}

// =====================================================================
section("부록 · 자기 점검");
{
  const s = slide("DARK", "부록", "자기 점검과 운영위원회 요청 항목", `
[노트] 부록은 발표 시간에 설명하지 않습니다. 사전심사용입니다. 상세 설명은 각 슬라이드의 노트에 있습니다.
- 표기: [정형 README] = 최종 검증 자료 팜모니_정형/README.md, [이미지 README] = 팜모니_이미지/README.md, [이미지 data] = 팜모니_이미지/data/README_data.md
- 해당하지 않으면 '해당 없음', 하지 않은 확인은 '미실시', 알 수 없는 것은 '미확인'으로 적었습니다.`);
  const items = [["A1", "정형 · 자기 점검 1·4·5·6"], ["A2", "정형 · 자기 점검 2·3 (공통 1)"], ["A3", "이미지 · 자기 점검 1·4·5·6"], ["A4", "이미지 · 자기 점검 2·3 (공통 1, 팀별 2)"], ["A5", "성능 수치의 출처 (공통 2)"], ["A6", "이미지 외부 처리와 이용조건 (공통 3, 팀별 3)"], ["A7", "정형 제출 이력 26회"], ["A8", "정형 부품별 채택 경위"]];
  items.forEach(([k, t], i) => {
    const col = i < 4 ? 0 : 1, row = i % 4, x = M + col * 6.2, y = TOP + 0.2 + row * 1.15;
    tb(s, k, { x, y, w: 0.9, h: 0.8, fontSize: 24, bold: true, color: ACC, valign: "middle" });
    tb(s, t, { x: x + 1.0, y, w: 5.0, h: 0.8, fontSize: 16, color: BG, valign: "middle" });
  });
}
function grid4(s, cells) { // 2x2 카드
  const cw = (CW - 0.4) / 2, ch = 2.35;
  cells.forEach(([h, d], i) => {
    const x = M + (i % 2) * (cw + 0.4), y = TOP + Math.floor(i / 2) * (ch + 0.25);
    rbox(s, x, y, cw, ch, i === 0 ? INK : SUP);
    tb(s, h, { x: x + 0.25, y: y + 0.15, w: cw - 0.5, h: 0.4, fontSize: 14, bold: true, color: i === 0 ? ACC : INK });
    tb(s, d, { x: x + 0.25, y: y + 0.6, w: cw - 0.5, h: ch - 0.75, fontSize: 14, color: i === 0 ? BG : INK });
  });
}
{ // A1
  const s = slide("LIGHT", "부록 A1 · 정형", "자기 점검 1 · 4 · 5 · 6", `
1. 최종 검증 코드와 같은가: 같음. checks_v35e.py produce 결과가 24회차 업로드본과 바이트 단위로 같고 check ALL PASS. 전체 재학습 시 HR5 TabPFN을 GPU 대신 CPU로 돌려 EC 최대 .013 차이. [정형 README ⑥]
4. 공개 리더보드 사용처
- 제출물 비교·선택: 여러 부품을 제출 비교로 채택(부록 A8, [정형 README ⑤ 표 '최종 구성에 들어간 경위' 열]).
- 상수 결정: TabPFN 비중 0.4, 1건.
- (팀별 1) 0.3·0.4 고른 방법: 7회차 submission_09.csv(10-01)는 따뜻한 행 TabPFN 몫 0.3, 8회차 submission_11.csv(10-03)는 0.4. 두 제출 사이에 온도 열에서 바뀐 것은 이 비중 하나(제출/08회차 설명자료 표). 두 제출의 공개 온도 점수(.5193 대 .5160)를 비교해 0.4를 유지. 두 후보는 검증으로 좁힌 것: 비중을 올릴수록 추운 날이 나빠져(0.6에서 뚜렷) 0.4를 위쪽 한계로 봤고, 0.3과 0.4는 검증 우열이 엇갈렸음. 점수로 값을 계산하지는 않음.
5. 외부 데이터 없음. TabPFN v2(Prior Labs; 패키지 Apache-2.0; 가중치는 HF 저장소 Prior-Labs/TabPFN-v2-reg 표기를 따르며 원문 대조는 미실시; SHA256 2ab5a07d…). 코딩에 Claude·Codex. [정형 README ①]
6. 팀이 판단·설계: 문제 정의, 채택 기준(실행 전 고정, 모든 검증기 같은 방향), 규정 경계('예측 시점 이전 정보까지만'), W2 제거·현장형 정리, 제출 후보와 최종본 선택. AI(Claude·Codex)가 수행: 코드 작성, 실험 실행, 가설·실험 제안, 문서 초안.`);
  grid4(s, [["1. 최종 검증 코드와 같은가", "같음 — produce 결과가 24회차 업로드본과 바이트 단위로 같음, check ALL PASS"],
            ["4. 공개 리더보드 사용처", "제출물 비교·선택(부품 채택, 부록 A8) · 상수 결정 1건: TabPFN 비중 0.4(0.3·0.4 두 제출의 공개 온도 점수 .5193 대 .5160 비교)"],
            ["5. 외부 자료·사전학습·생성형 AI", "외부 데이터 없음 · TabPFN v2(Prior Labs) · 코딩에 Claude·Codex"],
            ["6. AI가 한 것 / 팀이 한 것", "팀: 문제 정의·채택 기준·규정 경계·W2 제거·최종 선택\nAI: 코드·실험 실행·가설 제안·문서 초안"]]);
}
{ // A2
  const s = slide("LIGHT", "부록 A2 · 정형", "자기 점검 2 · 3 — 현장에서 바뀌는 처리", `
2. 예측 시점 정보만 쓰는가: 평가 행은 같은 온실의 그 시각까지 입력과 앞 번호 기록만 씁니다. 미래 입력 절단 3곳·다른 온실 격리 2개 검사 ALL PASS. 학습 자료(train_X·train_y)는 10/5 중간 안내에 따라 날짜 제한 없이 씁니다. [정형 README ③]
3. 규칙성 의존 부분과 현장에서 바뀌는 처리: 표 참고.
- 이 처리를 뺀 성능(ABL1, 11-02 추가 실시, 서술용, 실행 전 고정 커밋 45ac69c5): 어제 이어 붙이기(앵커·사슬)를 끄면 학습 자료 검증기 2차 행 RMSE가 온도 +3.1~+10.4%, EC +8.3~+30.5% 나빠집니다. 온도 몫 중 사슬(현장에서도 가능)은 0.4~3.1%p. 상세: 집/클로드/발표/ABL1_어제이어붙이기_끄기_결과_2026-11-02.md
- 계절 지표를 뺀 성능: 미실시. 현장형 사슬(실제 어제) 성능: 미실시.
- 학습 자료와 출처가 다른 자료로 확인: 예측 사슬 식은 다른 49개 온실에서 외부 확인(48곳 개선). [정형 README ④] 나머지 부품은 미실시.
- excl 특징: 링커 반사실 검사에서 excl 가중치를 0으로 두면 앵커 15기록 중 4기록이 확신 .8 아래로 내려감. 학습 정답의 이어짐만 보지만 평가 쪽 기록을 간접적으로 가려낼 수 있다는 점을 밝혀 둠. [정형 README ③]`);
  table(s, [["처리", "기대는 대회 자료의 규칙성", "현장에서는"],
            ["FL 링커", "하루 기록이 섞이고 같은 동이 두 ID를 오감", "필요 없음 — 달력으로 어제가 정해짐"],
            ["어제 정답 앵커 (23기록)", "평가 기록 사이에 정답 있는 학습 기록이 끼어 있음", "전날 배지 실측이 있을 때만 가능"],
            ["예측 사슬 (17기록)", "앞선 평가 기록이 어제로 연결됨", "가능 — 어제의 내 예측을 이어 씀"],
            ["excl 특징", "학습 정답의 이어짐", "필요 없음"],
            ["계절 지표", "2차 구간 외기가 1차 구간 어떤 날과 거의 같음", "달력 날짜로 대체"],
            ["NX · HR5 · TX3", "해당 없음(같은 기록의 그 시각까지)", "그대로"]],
        { x: M, y: TOP, w: CW, colW: [2.6, 5.0, 4.53], rowH: 0.5, fontSize: 14 });
  rbox(s, M, TOP + 3.75, CW, 1.0, INK);
  tb(s, "끄기 실험(ABL1): 앵커·사슬을 끄면 검증기 2차 행 RMSE 온도 +3.1~+10.4%, EC +8.3~+30.5% · 계절 지표를 뺀 성능은 미실시", { x: M + 0.3, y: TOP + 3.75, w: CW - 0.6, h: 1.0, fontSize: 14, bold: true, color: BG, valign: "middle" });
}
{ // A3
  const s = slide("LIGHT", "부록 A3 · 이미지", "자기 점검 1 · 4 · 5 · 6", `
1. 같음. 제출 ZIP의 추론 코드 그대로이며, 보관 ZIP을 CPU로 재추론해 공개 21장 21/21, 정순·역순 결과 차 0. [이미지 README ⑥·⑧]
4. 제출물 비교·선택에 사용: 나단9와 나단11 중 최종을 고를 때 환산점수(23.82 → 25.59)를 비교. 사전 규칙(LB Accuracy ≥ .98)과 다르게 정한 이유: Accuracy가 소수 둘째 자리까지만 표시돼 두 판이 같게 보였고, 환산점수와 내부 검증(GPT 가짜를 더 잡고 실제 오판은 늘지 않음)에서 나단11이 나았음. 상수·문턱 결정에는 쓰지 않음. [이미지 README ⑤ 끝]
5. 부록 A6.
6. 팀이 판단·설계: 문제 정의, 채택 기준, AI Hub 데이터셋 선택과 이용 신청, 제출 후보와 최종본 선택. AI(Claude·Codex)가 수행: 코드 작성, 실험 실행, 가설·실험 제안, GPT 이미지 생성 작업, 문서 초안.`);
  grid4(s, [["1. 최종 검증 코드와 같은가", "같음 — 제출 ZIP 추론 코드 그대로, CPU 재추론 공개 21/21"],
            ["4. 공개 리더보드 사용처", "제출물 비교·선택: 나단9 vs 나단11 환산점수(23.82 → 25.59) · 상수·문턱에는 안 씀"],
            ["5. 외부 자료·사전학습·생성형 AI", "AI Hub 596 · CLIP(MIT) · SD·SDXL·Qwen · ChatGPT 이미지 — 부록 A6"],
            ["6. AI가 한 것 / 팀이 한 것", "팀: 문제 정의·채택 기준·AI Hub 선택·최종 선택\nAI: 코드·실험·GPT 생성 작업·문서 초안"]]);
}
{ // A4
  const s = slide("LIGHT", "부록 A4 · 이미지", "자기 점검 2 · 3 — 원본 대조와 형식 조건이 작동하는 입력", `
2. 이미지 한 장만 입력으로 씁니다. 이미지끼리 정보를 주고받지 않고, 순서 독립 검사 OK(공식 self_check, 10-16). GPU 8장 묶음 계산에 따른 수치 차는 최대 .008. [이미지 README ③]
(팀별 2) 각 분기가 작동하는 입력과 작동하지 않을 때의 처리 — 표. [이미지 README ⑤ 표·정의]
- 정밀 지문: 덮게 축소·가운데 768×1024 크롭 후 흑백 24×32 평균 g와 64px 구역 12×16 라플라시안 분산 log값 e. DB 1위 거리 dg1, gap = 2위 − 1위. 국소 차이 lg·le가 크면 가짜 확정(p ← max(p, .95)), 작으면 진짜 확정(p < .8일 때 p ← min(p, .3)).
- 예외: 원본 대조 계산이 실패하면 R1·정밀 대조 규칙을 적용하지 않음.
현장에서 어떻게 바뀌나: 현장 사진은 대부분 AI Hub DB에 원본이 없으므로 원본 대조 두 단계가 거의 작동하지 않아 '지문 DB를 끈 판'과 비슷해집니다. 지문 DB 끄기 실험: 균형 정확도 R2~R5 −1.4~−2.3%p(README 반올림 수치 기준), 주로 실제 오탐 증가. 현장 사진이 다른 형식(PNG, 다른 JPEG 품질)이면 형식 게이트가 꺼져 전문가 3개가 빠집니다 — 형식 게이트를 끈 성능은 미실시.
학습 자료와 출처가 다른 자료로 확인: 미실시. 봉인 R2~R5는 같은 AI Hub 데이터셋에서 촬영 묶음만 다르게 뗀 것. 공개 21장은 운영진 자료지만 설계·선택에 썼음.`);
  table(s, [["분기", "작동하는 입력", "작동하지 않는 사진"],
            ["R1 색 지문 대조", "DB 1위 거리 3.5 < d1 < 20, gap > 8, p ≥ .25 → p를 .9 이상으로", "기본 판별 확률 그대로"],
            ["정밀 원본 대조", "dg1 ≤ 1.011, gap ≥ 3.474(원본이 DB에 있음) → 국소 차이 크면 가짜 확정, 거의 같으면 진짜 확정", "건너뜀 → 기본 판별 + 전문가만"],
            ["형식 게이트", "JPEG · 4:2:0 · PIL Q92 양자화표와 완전히 같음 → 전문가 H · E_gr · E_gpt 적용", "이 세 전문가 미적용"]],
        { x: M, y: TOP, w: CW, colW: [2.3, 6.3, 3.53], rowH: 0.8, fontSize: 14 });
  rbox(s, M, TOP + 3.55, CW, 1.2, INK);
  tb(s, "현장 사진은 대부분 DB에 원본이 없어 원본 대조가 작동하지 않음 → 지문 DB 끈 판에 가까움(균형 정확도 −1.4~−2.3%p) · 형식 게이트를 끈 성능은 미실시", { x: M + 0.3, y: TOP + 3.55, w: CW - 0.6, h: 1.2, fontSize: 14, bold: true, color: BG, valign: "middle" });
}
{ // A5
  const s = slide("LIGHT", "부록 A5 · 공통 2", "발표에 쓴 성능 수치의 출처", `
수치마다 낸 자료, 그 자료를 모델·설정 선택에도 썼는지, 학습 자료와 겹치는 범위를 표로 정리했습니다.
- 정형 검증기 값 선택 예: HR5 λ 0.5는 DIAG10에서 판정 전에 선택, EC 앵커 강도 0.8·p는 검증 90조합 중 선택, TAG2·CHT 선택지는 검증 블록에서 데이터로 선택. [정형 README ⑤ 표]
- 이미지 문턱 θ의 실제 풀: 학습 실제 묶음 5폴드 OOF + 학습에 안 쓴 실제 1,395장.
- 이미지 봉인 R2~R5: 학습·문턱 계산에는 쓰지 않고 후보 채택 판정 때만 열었으며, 10-20 지문 DB 끄기 실험(서술용)에서 다시 썼습니다.
- 공개 21장: 10-10~10-12 초기 후보 구성(가짜 제작 5판 포함 여부, 기본 판별기 조합)을 고를 때 지표로 5회 이상 사용, 이후 모든 후보에서 21/21 유지 점검. [이미지 data 4절]`);
  table(s, [["수치", "낸 자료", "선택에도 썼나", "학습과 겹침"],
            ["정형 공개 점수 .4608 · .1148", "플랫폼 평가 자료", "예 — 부품 채택, TabPFN 0.4", "없음"],
            ["정형 검증기 값", "F13·F47 학습 기록을 기록 단위로 가림", "예 — 채택 판정, 일부 값 선택", "같은 학습 자료 안에서 나눔"],
            ["정형 ABL1 끄기 결과", "위 검증기", "아니오(서술용)", "같은 학습 자료"],
            ["이미지 공개 점수 .97 · 25.59", "운영진 시험 이미지", "예 — 회차 채택, 최종 선택", "없음"],
            ["이미지 봉인 R2~R5", "AI Hub 596, 촬영 묶음이 학습과 안 겹침", "예 — 후보 채택 판정", "같은 데이터셋·제작법, 원본 사진은 안 겹침"],
            ["이미지 공개 21장", "운영진 공개 샘플", "예 — 초기 구성 선택, 21/21 점검", "학습에는 안 넣음"],
            ["이미지 문턱 θ", "실제 풀(학습 실제 OOF + 미사용 실제)", "예 — 문턱 자체", "학습 실제와 겹침(OOF)"]],
        { x: M, y: TOP, w: CW, colW: [2.9, 3.6, 2.9, 2.73], rowH: 0.58, fontSize: 12 });
}
{ // A6-1
  const s = slide("LIGHT", "부록 A6 · 공통 3", "이미지 자료의 외부 처리 ① ②", `
① 외부 생성형 서비스에 공개 데이터 사진을 넣었나: 예.
- 서비스: ChatGPT 내장 이미지 생성·편집(OpenAI Codex 도우미로 수행).
- 데이터셋: AI Hub 「지능형 수직농장 통합 데이터(딸기)」(596).
- 입력 범위: 서로 다른 AI Hub 원본 525장(gpt1 75 · gpt2 150 · gpt3 150 · gptr 150, 묶음 사이 겹침 0). 11-02 기록 검산(Drive lab_claude_local/im_set_gpt1~3/manifest.csv의 source 열, data/gpt_generation/codex_gpt_regen_20261015/목록.csv).
- 만든 이미지 600장: gpt1 25장은 원본 없이 문장만으로 생성. gpt3 200장 가운데 '전체 다시 그리기' 50장은 같은 원본 50장을 편집하는 과정의 1단계 결과물(최종 검증 자료의 manifest_v3.csv 150행에는 이 50장이 빠져 있음).
- 최종 학습 범위: E_gpt 학습에 600장 모두, E_pg·E_ps·E_L 학습에 gpt1·gpt2 250장. [이미지 README ①·④]
② 원격 환경에서의 처리·보관
- Google Colab: 정밀 지문 DB 생성(AI Hub 원천 사진 처리), 추론 시간 측정. 처리 국가 미확인.
- 연구실 GPU 서버(ADILAB, RTX 6000 Ada): 가짜 생성·특징 계산·학습. 학교 안 서버, 국내(팀 확인). 생성·학습에 쓴 이미지 세트가 이 서버에 있음.
- Google Drive(farmai_sync, 팀원 개인 계정): 연구실 PC와 집 노트북 동기화용. AI Hub에서 고른 원본 사진(im_real_raw 등), GPT 세트, tar별 지문 파일을 보관(11-02 폴더 확인). 보관 국가 미확인.`);
  const cw = (CW - 0.4) / 2;
  rbox(s, M, TOP, cw, 4.8, SUP);
  tb(s, "① 외부 생성형 서비스 입력", { x: M + 0.3, y: TOP + 0.2, w: cw - 0.6, h: 0.4, fontSize: 16, bold: true });
  stat(s, M + 0.3, TOP + 0.5, cw - 0.6, "525장", "ChatGPT에 넣은 서로 다른 AI Hub 원본", INK, 40);
  bullets(s, ["서비스: ChatGPT 내장 이미지 생성·편집", "묶음: gpt1 75 · gpt2 150 · gpt3 150 · gptr 150", "만든 이미지 600장(25장은 문장만으로)", "최종 학습: E_gpt 600장 · E_pg·E_ps·E_L 250장"], { x: M + 0.3, y: TOP + 2.2, w: cw - 0.6, h: 2.5, fontSize: 14 });
  const x2 = M + cw + 0.4;
  tb(s, "② 원격 처리·보관", { x: x2, y: TOP + 0.2, w: cw, h: 0.4, fontSize: 16, bold: true });
  table(s, [["환경", "단계", "국가"], ["Google Colab", "정밀 지문 DB 생성, 시간 측정", "미확인"], ["연구실 GPU 서버", "가짜 생성 · 특징 · 학습", "국내(학교)"], ["Google Drive", "원본·GPT 세트·지문 파일 보관(동기화)", "미확인"]],
        { x: x2, y: TOP + 0.75, w: cw, colW: [1.7, 2.95, 1.31], rowH: 0.8, fontSize: 14 });
}
{ // A6-2
  const s = slide("LIGHT", "부록 A6 · 공통 3", "이미지 자료의 이용조건 ③ ④ — 지문 DB 두 개", `
③ 이용조건 확인·판단
- AI Hub: 이용 신청·승인 후 사용. 원본 재배포 제한은 일반 지식 수준으로 판단했고 약관 원문 대조는 미실시. 그래서 원본과 파생 이미지 16,197장은 최종 검증 자료에 파일로 넣지 않고 원본 대응 해시 표와 문서로 냈음(10-20). 이용조건 때문에 바꾼 결정은 이것 하나(팀 확인).
- 생성 모델 라이선스: 모델 카드 기준 일반 지식(SD1.5 CreativeML OpenRAIL-M, SDXL OpenRAIL++-M, sdxl-vae-fp16-fix MIT, Qwen-Image-Edit Apache-2.0). ChatGPT: OpenAI 이용약관 기준 일반 지식. [이미지 data 3절]
- 출처를 확인하지 못한 자료: 해당 없음. 모든 학습 이미지에 원본 경로나 생성 기록이 있음(data/source_sha256.csv, data/generated_images/).
④ (팀별 3) 제출 모델의 지문 DB 두 개
- fp_db.npz.xz: 사진별 12×16 RGB 평균(576값, 사진당 576바이트).
- finefp_db.npz.xz: 사진별 흑백 24×32 평균(768값) + 64px 구역 12×16 라플라시안 분산 log(192값).
- 원천: AI Hub 596 원천 사진 152,043장 전체(Training 271 tar + Validation 35 tar).
- 넣은 판단: 사진이 아니라 사진별 숫자 값이라 출처를 밝히고 넣음(팀 결정 2026-10-12). AI Hub 약관 원문 대조는 미실시. [이미지 data 1절]
- 축소본 성격을 밝힘(팀 결정 11-02): 두 DB의 값은 사실상 사진의 아주 작은 축소본(12×16 컬러, 24×32 흑백)으로, 원 사진(768×1024) 대비 화소 수 1/4,096, 1/1,024. 원 사진을 대신할 해상도는 아니지만 사진별 값을 담고 있으므로 공통 3 ④ 대상으로 보고 적음.`);
  table(s, [["DB", "사진별로 담은 값", "크기", "원천"],
            ["fp_db", "12×16 RGB 평균 — 아주 작은 컬러 축소본", "576값(576바이트)", "AI Hub 596 원천 152,043장 전체"],
            ["finefp_db", "흑백 24×32 평균 — 아주 작은 흑백 축소본 + 12×16 라플라시안 에너지", "768 + 192값", "같은 152,043장"]],
        { x: M, y: TOP, w: CW, colW: [1.6, 5.4, 2.2, 2.93], rowH: 0.75, fontSize: 14 });
  const cw = (CW - 0.4) / 2;
  rbox(s, M, TOP + 2.55, cw, 2.2, SUP);
  tb(s, [{ text: "③ 이용조건 판단", options: { bold: true, breakLine: true } }, { text: "AI Hub 이용 신청·승인 · 재배포 제한 때문에 원본·파생 이미지 16,197장은 해시 표로 대체(10-20) · 약관 원문 대조는 미실시" }], { x: M + 0.3, y: TOP + 2.7, w: cw - 0.6, h: 1.9, fontSize: 14, paraSpaceAfter: 6 });
  rbox(s, M + cw + 0.4, TOP + 2.55, cw, 2.2, INK);
  tb(s, [{ text: "④ 제출 모델에 넣은 판단", options: { bold: true, color: ACC, breakLine: true } }, { text: "출처를 밝히고 포함(10-12) · 원 사진 대비 화소 수 1/4,096 · 1/1,024의 축소본 성격임을 밝힘", options: { color: BG } }], { x: M + cw + 0.7, y: TOP + 2.7, w: cw - 0.6, h: 1.9, fontSize: 14, paraSpaceAfter: 6 });
}
{ // A7 제출 이력
  const hist = [[1, "09-22", ".7450", ".2442", "기준 모델"], [2, "09-25", ".6666", ".2287", "특징 축소·모델 혼합"], [3, "09-26", ".5624", ".2055", "물리 기준선"], [4, "09-26", ".5697", ".2055", "한랭 보정(악화)"], [5, "09-26", ".5456", ".2134", "학습 행 가중 0.2"], [6, "09-30", ".5251", ".2055", "실내온도 게이트 섞기"], [7, "10-01", ".5193", ".2774", "TabPFN 0.3"], [8, "10-03", ".5160", ".1455", "TabPFN 0.4 · EC 계절 지표"], [9, "10-06", ".5127", ".1384", "계절 지표 · SG2"], [10, "10-11", ".5126", ".3538", "GRU · EC 고EC 날 제외(악화)"], [11, "10-11", ".5089", ".1305", "새벽 보정 · W2 포함"], [12, "10-11", ".5012", ".1271", "앵커 첫 판 · W2 포함"], [13, "10-12", ".4883", ".1279", "앵커 확장 · W2 포함"],
                [14, "10-13", ".4864", ".1077", "일치 앵커 · B5 · W2 포함"], [15, "10-14", ".4853", ".0979", "HR5 · W2 포함"], [16, "10-15", ".4861", ".1225", "W2·SG2·B5 제거"], [17, "10-15", ".4817", ".1170", "현장형 구성"], [18, "10-15", ".4922", ".1173", "GRU 30%(악화)"], [19, "10-16", ".4706", ".1153", "예측 사슬"], [20, "10-16", ".4754", ".1153", "이력 재연결(악화)"], [21, "10-16", ".4685", ".1153", "사슬 문턱 .5"], [22, "10-16", ".5000", ".1153", "범위 밖 보정(악화)"], [23, "10-16", ".4665", ".1153", "TAG2 앵커"], [24, "10-16", ".4608", ".1148", "CHT 사슬 · EC TAG2"], [25, "10-16", ".4845", ".1148", "실험(되돌림)"], [26, "10-16", ".4608", ".1148", "24회차 재제출(최종)"]];
  [[0, 13, "1~13회차"], [13, 26, "14~26회차"]].forEach(([a, b, t]) => {
    const s = slide("LIGHT", "부록 A7 · 정형", "제출 이력 " + t, `
팀 회차 = 플랫폼 제출 번호(팀원 확인, 10-20). 공개 점수는 낮을수록 좋음(RMSE). 제출마다 미리 만든 후보 구성 하나를 냈고, 공개 점수로는 '다음 후보의 바탕으로 유지할지'만 정했습니다. W2 규칙은 11~15회차에 있었고 16회차부터 뺐습니다(주최 측 10/16 답변에서 금지 확인). 6회차는 팀원 개인 몫 제출. 출처: 제출/제출이력_정형_2026-10-20.md`);
    table(s, [["회차", "날짜", "온도", "EC", "바꾼 것"]].concat(hist.slice(a, b)), { x: M, y: TOP, w: CW, colW: [1.0, 1.3, 1.3, 1.3, 7.23], rowH: 0.34, fontSize: 12 });
  });
}
{ // A8 부품별 채택 경위
  const s = slide("LIGHT", "부록 A8 · 정형", "부품별로 값을 정한 방법과 채택 경위", `
'로컬 판정' = 실행 전에 고정한 기준(모든 검증기·시드 같은 방향 + P < .025)으로 검증기 결과를 판정한 것. '제출 비교로 채택' = 로컬 기준을 다 채우지 못한 구성을 후보로 제출했고, 공개 점수가 직전 최고보다 나아져 다음 구성의 바탕으로 유지한 경우. 공개 점수로 가중치·계수·문턱을 계산하지 않았음(예외: TabPFN 비중 0.4는 미리 정한 두 값 중 선택). 출처: [정형 README ⑤ 표]
세부: NX 0.4는 DIAG10 −0.8%이나 추운 검증 +0.1~0.2%(기준 미달) / 계절 지표 2차 DIAG10 −1.35%(탐색) / day·second 제거는 혼재 / 이전 판 앵커 P .093 / TAG2 우월 P .12, 대체 판정 통과 / HR5 P2LOO 2차 −9.1%·EL1 −4.1%이나 이득이 고EC 인접 기록에 집중 / EC 앵커 P .35, 방향 6/6 / EC TAG2 P .0026이나 DIAG10 한 기록 손해(미달).`);
  table(s, [["부품", "값을 정한 방법", "로컬 판정", "채택 경위"],
            ["NX 섞기 0.4/0.2/0.4", "검증으로 0.3·0.4로 좁힘", "기준 미달", "제출 비교(9·11번 파일)"],
            ["계절 지표", "검증 탐색", "탐색", "제출 비교"],
            ["day·second 제거", "검증기 4종 비교", "혼재", "현장형 정리로 채택"],
            ["FL 링커 · 어제 앵커", "학습 정답으로 적합", "P .093(미달)", "제출 비교(현장형 구성)"],
            ["TAG2 앵커 확장", "검증 블록에서 데이터 선택", "대체 판정 통과", "제출 비교"],
            ["TX3 0.25/0.8/10℃", "학습 정답 측정값", "통과 P .0047", "로컬 채택"],
            ["예측 사슬 · 문턱 .5", "검증 사슬 행에 적합", "통과 P .0003 · .0029", "로컬 채택"],
            ["CHT 사슬 확장", "검증 블록에서 선택", "대체 판정 통과", "제출 비교(최종)"],
            ["HR5 λ 0.5", "DIAG10에서 판정 전 선택", "고EC 인접에 집중", "제출 비교"],
            ["EC 앵커 0.8·p · EC TAG2", "검증 90조합 중 선택 · TAG2 조건", "P .35 · .0026(미달)", "제출 비교(최종)"]],
        { x: M, y: TOP, w: CW, colW: [3.0, 3.6, 2.6, 2.93], rowH: 0.44, fontSize: 12 });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT, "slides", n);
})();
