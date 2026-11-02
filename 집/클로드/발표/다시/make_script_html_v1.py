# 발표 대본을 슬라이드 그림과 나란히 보는 HTML로 만든다 (2026-11-03, 집 클로드)
# 입력: 덱 v4를 한쇼로 뽑은 PNG(s01~s16), 출력: 발표대본_v2_2026-11-03.html (그림을 파일 안에 넣은 한 파일)
import base64, io, sys
from pathlib import Path
from PIL import Image

PNG = Path(sys.argv[1])  # 슬라이드 PNG 폴더
OUT = Path(__file__).with_name("발표대본_v2_2026-11-03.html")

# (장 번호, 제목, 시간(초), 핵심 한 줄, [대본 단락], [가리킬 곳])
S = [
    (1, "표지", 15, "현장에서 그 시각에 알 수 있는 정보로 예측하고 판별한다",
     ["안녕하십니까, 팜모니 AI 발표를 맡은 조나단입니다.",
      "저희는 정형과 이미지 두 미션을 <b>하나의 원칙</b>으로 풀었습니다.",
      "<b>현장에서 그 시각에 알 수 있는 정보로</b> 예측하고 판별한다는 원칙입니다."],
     []),
    (2, "결과 요약", 40, "정형 .4608 / .1148, 이미지 정확도 97%",
     ["결과부터 말씀드리겠습니다.",
      "정형은 온도 <b>.4608</b>, EC <b>.1148</b>. 두 목표 모두 저희 최고 기록이 최종 답안입니다.",
      "이미지는 운영진 채점에서 정확도 <b>97%</b>를 받았습니다.",
      "이 숫자를 어떻게 만들었는지, 그리고 <b>현장에서도 통하는지</b> 순서대로 보여 드리겠습니다."],
     ["왼쪽 정형 숫자 → 오른쪽 이미지 숫자"]),
    (3, "제1부 정형", 5, "", ["먼저 정형 미션입니다."], []),
    (4, "1.1 문제와 자료", 45, "기록이 섞여 있고, 같은 동이 두 ID를 오간다",
     ["과제는 두 온실의 배지 온도와 EC를 <b>시간별로</b> 예측하는 것입니다.",
      "자료를 들여다보니 두 가지가 보였습니다. 하루 기록이 <b>번호 순서와 상관없이 섞여</b> 있었고,",
      "같은 동이 <b>F13과 F47 두 ID를 오가고</b> 있었습니다. 확실한 짝의 약 4분의 1입니다.",
      "그래서 오늘의 <b>'어제'가 어디 있는지 찾는 것</b>이 열쇠가 됐습니다."],
     ["왼쪽 표: 과제", "오른쪽 그림: 4일의 어제(3일)가 다른 줄에 있음"]),
    (5, "1.2 핵심 발견", 45, "하루 통째로 어긋나는 오차의 답은 '어제'에 있다",
     ["오차를 나눠 보니 절반 가까이(<b>.49</b>)가 하루 전체가 통째로 어긋나는 몫이었습니다.",
      "이 몫은 그날 입력으로는 <b>설명되지 않았고</b>(R² 음수),",
      "대신 <b>전날 오차와 .64로</b> 강하게 이어져 있었습니다. 다른 온실들에서도 같았습니다.",
      "그래서 처방은 하나, <b>같은 동의 진짜 어제를 잇는 것</b>입니다."],
     ["숫자 세 개를 왼쪽부터 차례로", "아래 화살표 문장"]),
    (6, "1.3 모델 구성", 50, "기본 예측 → 어제 이어 붙이기 → 추운 시간 보정",
     ["모델은 <b>세 단계</b>입니다.",
      "① 세 모델을 섞어 기본 예측을 만들고, 8도 이하에서는 TabPFN을 뺍니다.",
      "② 핵심은 <b>어제 이어 붙이기</b>입니다. 어제 실측이 있으면 앵커로, 없으면 어제 예측을 이어 붙입니다.",
      "③ EC가 높은 날의 추운 시간대를 조금 보정합니다.",
      "아래 그래프처럼 어제는 <b>새벽에 크게</b> 반영되고, 낮에는 오늘 입력이 주도합니다."],
     ["위 흐름도 가운데 상자", "아래 그래프: 0시 높고 9시 무렵 낮음"]),
    (7, "1.4 현장형 정리", 45, "대회 표식은 빼고, 현장에도 있는 '어제'만 남겼다",
     ["마지막 단계에서 모델을 <b>현장형으로 정리</b>했습니다.",
      "기록 번호 자체, 2차 구간 표시처럼 <b>대회 자료에만 있는 표식은 모두 뺐습니다.</b>",
      "남긴 것은 현장에도 있는 <b>어제</b>입니다. 대회에선 찾아야 했지만 현장에선 <b>달력으로 정해집니다.</b>",
      "남은 한계는 계절 지표 하나이고, 현장에서는 달력 날짜로 바꾸면 됩니다."],
     ["표 위쪽 '뺀 것' → 가운데 '남긴 것' → 맨 아래 '남은 한계'"]),
    (8, "1.5 검증과 개선", 50, "사전 기준을 통과한 변경만 실제로 좋아졌다",
     ["검증은 학습 기록 일부를 가려 만든 <b>여러 검증 묶음</b>으로 했습니다.",
      "채택 기준은 <b>실행 전에 코드에 고정</b>했습니다. 모든 검증에서 같은 방향, 우연일 확률 2.5% 미만.",
      "기준을 통과한 <b>19·21회차</b>는 실제 점수도 좋아졌고,",
      "근소하게 못 넘은 <b>20·22회차</b>는 크게 나빠졌습니다.",
      "기준을 지키는 것 자체가 <b>실패를 거르는 장치</b>였습니다."],
     ["왼쪽 그래프: 20·22회차에서 튀어 오름", "오른쪽 표: '사전 기준' 열"]),
    (9, "1.6 현장 해석", 40, "어제를 끄면 오차가 온도 3~10%, EC 8~31% 늘어난다",
     ["어제 정보는 얼마나 중요할까요?",
      "끄고 일반 방법만 남기면 오차가 <b>온도 3~10%, EC 8~31%</b> 늘어납니다.",
      "배지 상태는 하루 단위로 이어집니다. <b>어제가 오늘의 가장 좋은 단서</b>입니다.",
      "그래서 현장에서는 <b>센서 기록이 끊기지 않는 것</b>이 정확도의 조건입니다."],
     ["왼쪽 온도 막대 → 오른쪽 EC 막대", "아래 '현장 조건' 문장"]),
    (10, "제2부 이미지", 5, "", ["다음은 이미지 미션입니다."], []),
    (11, "2.1 문제와 전략", 40, "정답 사진이 없어서 가짜를 직접 만들었다",
     ["이미지 미션에는 <b>정답이 붙은 학습 사진이 없었습니다.</b>",
      "그래서 공개 데이터의 실제 딸기 사진으로 <b>가짜를 직접 만들었습니다.</b> 필터, 생성, 편집까지요.",
      "검증도 저희가 만든 자료로 했습니다.",
      "그래서 이 절차는 <b>현장 사진만 있으면 그대로 다시 돌릴 수 있습니다.</b>"],
     ["위 흐름도: 왼쪽에서 오른쪽", "아래 표: 가짜 종류"]),
    (12, "2.2 모델 구성", 40, "사진 한 장만 보는 판별기, 대회용 장치는 따로",
     ["판별은 <b>사진 한 장만</b> 보고 합니다.",
      "기본 판별기가 먼저 판단하고, 놓치기 쉬운 유형은 <b>유형별 전문가</b>가 한 번 더 봅니다.",
      "아래 점선은 <b>대회 자료에서만 작동하는 장치 두 개</b>입니다. 지문 DB와 형식 게이트입니다.",
      "현장 사진에선 거의 꺼지기 때문에, 다음 장에서 <b>끄고 다시 쟀습니다.</b>"],
     ["위 상자 두 개", "아래 점선 상자"]),
    (13, "2.3 검증과 개선", 40, "학습에 안 쓴 봉인 묶음에서 94~96%",
     ["검증은 따로 떼어 둔 <b>봉인 묶음</b>으로 했습니다. 학습에 쓰지 않고, 후보를 고를 때만 열었습니다.",
      "원본 겹침이 없는 세 묶음에서 균형 정확도는 <b>94~96%</b>였습니다.",
      "실패에서 고친 것도 있습니다. GPT 편집 전문가의 첫 판이 기준을 근소하게 못 넘었고,",
      "원인을 찾아 <b>문제가 된 사진에는 쓰지 않도록</b> 고쳤습니다."],
     ["왼쪽 막대", "오른쪽 표: '원본 겹침' 열"]),
    (14, "2.4 현장 적용", 50, "장치를 꺼도 핵심은 유지, 약점은 휴대폰 사진",
     ["현장 사진에는 AI Hub 원본이 없으니 <b>지문 DB를 끄고 다시 쟀습니다.</b>",
      "<b>1.4~2.3%p</b> 내려가지만 핵심 성능은 유지됩니다.",
      "오른쪽은 솔직하게 말씀드릴 약점입니다. 선명화하면 <b>90.7%</b>, WebP로 저장하면 <b>37.5%</b>를 가짜로 판정합니다.",
      "그래서 현장에서는 <b>그 농장 사진으로 같은 절차를 다시 돌려</b> 확인해야 합니다."],
     ["왼쪽 막대: 진한 색 vs 연한 색", "오른쪽 가로 막대: 맨 위 '선명화'"]),
    (15, "함께 일한 방식", 25, "판단은 팀, 실행은 AI",
     ["AI 도구를 많이 썼습니다.",
      "다만 <b>무엇을 할지, 무엇을 채택할지, 규정 경계</b>는 팀이 정했고,",
      "AI는 코드와 실험 실행, 제안, 초안을 맡았습니다. <b>예측 단계에는 AI를 쓰지 않습니다.</b>"],
     ["표의 왼쪽 열 → 오른쪽 열"]),
    (16, "정리", 25, "한 것과 아직 하지 않은 것",
     ["정리하겠습니다. 가운데가 <b>한 것</b>, 오른쪽이 <b>아직 하지 않은 계획</b>입니다.",
      "저희는 높은 점수보다, <b>그 점수가 현장에서도 통하는지</b>를 끝까지 확인하려고 했습니다.",
      "감사합니다."],
     []),
]

QA = [
    ("TabPFN 비중 0.4는 어떻게 골랐나?", "내부 검증이 엇갈려 공개 리더보드 비교 한 번으로 정했다. 우연 가능성은 배제 못 함", "A7"),
    ("어제 찾기는 대회 자료 요령 아닌가?", "대회는 기록이 섞여 찾아야 했고, 현장은 달력으로 정해진다. 끈 성능도 쟀다", "1.4 · 1.6 · A3"),
    ("리더보드는 어디에 썼나?", "제출물 비교·선택. 상수 결정은 TabPFN 0.4 한 건", "A1"),
    ("지문 DB는 무엇이고 넣어도 되나?", "사진별 아주 작은 축소값. 출처를 밝혀 포함, AI Hub 문의는 미실시", "A6"),
    ("형식 게이트도 끄면?", "R3~R5 결과 반영 예정. R2에서는 두 장치 모두 끄면 .964 → .939", "2.4 · A8"),
    ("봉인 묶음이 학습과 겹치지 않나?", "사진은 안 겹치지만 농장·촬영 묶음은 겹친다. README 문구 정정", "A4"),
    ("외부 AI에 사진을 넣었나?", "ChatGPT에 525장. 처리 국가는 미확인", "A5 · A6"),
    ("처음 보는 온실에서도 되나?", "미실시. AI Hub 44농장을 모두 학습에 썼다", "2.4"),
]


def thumb(i):
    im = Image.open(PNG / f"s{i:02d}.png").convert("RGB")
    im.thumbnail((720, 405))
    b = io.BytesIO(); im.save(b, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def mmss(t):
    return f"{t // 60}:{t % 60:02d}"


cum = 0
cards = []
for no, title, sec, key, paras, points in S:
    cum += sec
    part = sec <= 5
    body = "".join(f"<p>{p}</p>" for p in paras)
    pts = "".join(f"<li>{p}</li>" for p in points)
    cards.append(f"""
<section class="card{' part' if part else ''}">
  <div class="shot"><img src="{thumb(no)}" alt="{no}번 장"></div>
  <div class="txt">
    <div class="head"><span class="no">{no}</span><h2>{title}</h2><span class="time">{mmss(sec)}</span><span class="cum">누적 {mmss(cum)}</span></div>
    {f'<div class="key">{key}</div>' if key else ''}
    <div class="say">{body}</div>
    {f'<div class="pt"><span>가리킬 곳</span><ul>{pts}</ul></div>' if pts else ''}
  </div>
</section>""")

rows = "".join(f"<tr><td>{q}</td><td>{a}</td><td class='c'>{p}</td></tr>" for q, a, p in QA)
overview = "".join(f"<tr><td class='c'>{no}</td><td>{t}</td><td class='c'>{mmss(s)}</td><td>{k}</td></tr>" for no, t, s, k, _, _ in S if s > 5)
html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>팜모니 AI 발표 대본</title>
<style>
:root {{ --ink:#2E2A24; --sub:#6B645A; --line:#E4DDD1; --olive:#5F6B2E; --ochre:#B98A3E; --bg:#FBFAF7; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"함초롬돋움","HCR Dotum","Malgun Gothic",sans-serif; font-size:17px; line-height:1.7; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:36px 20px 80px; }}
h1 {{ font-family:"함초롬바탕","HCR Batang",serif; font-size:30px; margin:0 0 6px; }}
.meta {{ color:var(--sub); margin-bottom:26px; }}
.meta b {{ color:var(--olive); }}
table {{ width:100%; border-collapse:collapse; background:#fff; margin:8px 0 34px; font-size:15px; }}
th {{ text-align:left; border-top:2px solid var(--ink); border-bottom:1px solid var(--ink); padding:8px 10px; }}
td {{ border-bottom:1px solid var(--line); padding:8px 10px; vertical-align:top; }}
tr:last-child td {{ border-bottom:2px solid var(--ink); }}
td.c {{ text-align:center; white-space:nowrap; color:var(--sub); }}
h3 {{ font-family:"함초롬바탕","HCR Batang",serif; font-size:21px; margin:34px 0 10px; }}
.card {{ display:grid; grid-template-columns:44% 1fr; gap:26px; background:#fff; border:1px solid var(--line); border-radius:6px; padding:20px; margin:16px 0; page-break-inside:avoid; }}
.card.part {{ grid-template-columns:22% 1fr; padding:12px 20px; }}
.shot img {{ width:100%; border:1px solid var(--line); display:block; }}
.head {{ display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }}
.no {{ font-family:"함초롬바탕","HCR Batang",serif; color:var(--olive); font-weight:bold; font-size:22px; }}
h2 {{ font-family:"함초롬바탕","HCR Batang",serif; font-size:22px; margin:0; }}
.time {{ background:var(--olive); color:#fff; border-radius:3px; padding:0 8px; font-size:14px; }}
.cum {{ color:var(--sub); font-size:14px; }}
.key {{ margin:10px 0 8px; padding:6px 12px; border-left:4px solid var(--ochre); background:#F6F0E4; font-weight:bold; }}
.say p {{ margin:0 0 8px; }}
.say b {{ color:var(--olive); }}
.pt {{ margin-top:10px; font-size:14px; color:var(--sub); }}
.pt span {{ font-weight:bold; }}
.pt ul {{ margin:2px 0 0 18px; padding:0; }}
@media (max-width:760px) {{ .card, .card.part {{ grid-template-columns:1fr; }} }}
@media print {{ body {{ background:#fff; font-size:13px; }} .card {{ border:none; border-bottom:1px solid var(--line); border-radius:0; }} }}
</style></head><body><div class="wrap">
<h1>팜모니 AI 발표 대본</h1>
<div class="meta">덱 v4 기준 · 발표 조나단 · 목표 <b>9분 20초</b>(10분 이내) · 초록 굵은 글씨 = 힘주어 말할 곳</div>
<h3>한눈에 보는 순서</h3>
<table><tr><th style="width:44px">장</th><th style="width:160px">제목</th><th style="width:60px">시간</th><th>한 줄 요지</th></tr>{overview}</table>
<h3>장별 대본</h3>
{''.join(cards)}
<h3>질의응답 대비</h3>
<table><tr><th>예상 질문</th><th>짧은 답</th><th style="width:110px">열 장</th></tr>{rows}</table>
<p class="meta">리허설 기준: 9분 30초를 넘으면 1.3의 그래프 설명이나 2.3의 실패 사례를 줄입니다.</p>
</div></body></html>"""
OUT.write_text(html, encoding="utf-8")
print("wrote", OUT, round(OUT.stat().st_size / 1024), "KB")
