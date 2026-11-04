# 대본 v15 → v16: 집 최종 v5 덱(본문 16장)의 노트 [말할 것]으로 장별 대사를 다시 짜고, 질의응답 6문항을 더한다 (2026-11-04, 집 클로드)
# 대사가 v15와 같은 장은 v15 절(굵은 표시·가리킬 곳 포함)을 그대로 쓰고, 바뀐 장은 노트 대사로 새로 쓴다. (변수 이름 v14 = 원본 대본)
import re
import sys

from pptx import Presentation

SRC = "집/클로드/발표/최종/발표대본_집v5_최종_v16_2026-11-04.md"
OUT = "집/클로드/발표/최종/발표대본_집v6_최종_v17_2026-11-04.md"
DECK = "집/클로드/발표/최종/팜모니_발표자료_집v6.pptx"
NBODY = 15

v14 = open(SRC, encoding="utf-8").read()

# v14 장별 절
sec = {}
for blk in re.split(r"\n(?=## \d+\. )", v14):
    m = re.match(r"## (\d+)\. ([^\n]*)\n(.*)", blk, re.S)
    if not m:
        continue
    body = m.group(3).split("\n## ")[0].split("\n---")[0]
    said = " ".join(l.strip() for l in body.split("\n") if l.strip() and not l.startswith("- ")).replace("**", "")
    sec[said] = body.rstrip()

prs = Presentation(DECK)
said_now, tags = [], []
for i, sl in enumerate(prs.slides, 1):
    if i > NBODY + 1:
        break
    note = sl.notes_slide.notes_text_frame.text if sl.has_notes_slide else ""
    m = re.search(r"\[말할 것\] (.*?)(\n|$)", note)
    said_now.append(m.group(1).strip() if m else "")
    texts = [sh.text_frame.text.strip() for sh in sl.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
    tags.append("표지" if i == 1 else texts[0])
need_n = NBODY + 1  # 표지 + 본문 15
said_now, tags = said_now[:need_n], tags[:need_n]


def secs(t):
    syl = len(re.findall(r"[가-힣]", t)) + len(re.findall(r"[0-9]", t)) * 0.8 + len(re.findall(r"[A-Za-z]", t)) * 0.4
    return syl / 3.54 + 1.5


def mmss(x):
    x = round(x)
    return f"{x // 60}:{x % 60:02d}"


parts, times, reused = [], [], 0
for i, (t, s_) in enumerate(zip(tags, said_now), 1):
    tm = secs(s_)
    times.append(tm)
    if s_ in sec:
        body = sec[s_]
        reused += 1
    else:
        body = s_
    parts.append(f"## {i}. {t} ({mmss(tm)})\n{body}\n")

total = sum(times)
order_lines = "\n".join(f"- {tags[i]} {mmss(times[i])}" for i in range(len(tags)))

# v15의 머리말과 본문 뒤 절(시간 맞추기, 질의응답 …)
head = v14.split("\n## 순서와 시간")[0]
tail = v14[v14.index("## 시간 맞추기"):]
tail = tail[: tail.index("## 대조 기록(집 v5")]
changes = []
tail = tail.replace("2. 1.2 (1/3)의 \"오른쪽 실제 하루처럼, 예측이 하루 내내 거의 같은 간격으로 낮습니다.\" (약 5초, 그림으로 보임)",
                    "2. 1.2 ①②의 \"아래 실제 하루처럼, 예측이 하루 내내 거의 같은 간격으로 낮습니다.\" (약 5초, 그림으로 보임)")
tail = tail.replace("· 1.2 (1/3) 노트", "· 1.2 노트")
QA17 = """
- **(집 v6) p < 0.025는 무슨 뜻인가?** 변경 전후 오차를 기록 묶음 단위로 여러 번 다시 뽑아(부트스트랩) 잰 '실제로는 나빠졌을 비율'이 2.5% 미만이라는 뜻입니다. 한쪽 p값과 같은 역할이고, 후보가 여럿이면 본페로니로 더 엄격하게 했습니다. · 1.6 노트
- **(집 v6) 세 검증은 언제부터 썼나?** 무작위·추운 날 검증은 9월 말부터, 채점 구간과 같은 배치 검증은 10월 중순부터입니다. 그래서 사전 기준 판정 색은 19~22회차에만 칠했습니다. · 1.6 노트
- **(집 v6) 추울 때 오차 그림의 모델은?** 찾은 점 ①과 같은 보정 전 기준 모델입니다. 모델 결정에 쓴 원래 측정(C만 쓰면 8℃ 이하 2.35~2.45℃, 고EC 동 +0.64℃)은 1.2 노트에 있습니다. · 1.2 노트
- **(집 v6) EC는 왜 하루 내내 같은 비율로 전날을 반영하나?** EC는 배지 속 양분이라 하루 안에서 천천히 바뀝니다. 시각별로 재 보니 14시까지는 전날 정보가 잘 맞았고, 채점 구간과 같은 배치 검증에서 하루 내내 반영하는 쪽이 나았습니다. 오후에는 과할 수 있어 다음에 다듬을 부분입니다. · 1.5 노트
"""
assert tail.count("\n### 나. 답이 약한 질문") == 1
tail = tail.replace("\n### 나. 답이 약한 질문", QA17 + "\n### 나. 답이 약한 질문", 1)

head = head.replace("# 팜모니 발표 대본 — 최종 v16 (2026-11-04, 집 클로드)",
                    "# 팜모니 발표 대본 — 최종 v17 (2026-11-04, 집 클로드)\n\n"
                    "- v16에서 바뀐 것: 기준 덱 **집 최종 v6**(`집/클로드/발표/최종/팜모니_발표자료_집v6.pptx`, 본문 15장). "
                    "찾은 점을 2장·3개로(③ 추울 때 오차), 1.5 대사 '두 모델을 섞어', 1.6 대사(검증 방식 + 회차 흐름), 질의응답 4문항 추가.")
head = re.sub(r"덱 본문은 1[0-9]장\.", "덱 본문은 15장.", head)
head = head.replace("- 기준 덱: `집/클로드/발표/최종/팜모니_발표자료_집v5.pptx` 본문 1~16장",
                    "- 기준 덱: `집/클로드/발표/최종/팜모니_발표자료_집v6.pptx` 본문 1~15장")
head = re.sub(r"- 길이: [^\n]*", f"- 길이: 음절 기준 추정 합계 {mmss(total)}(넘김 1.5초 포함, 리허설 전). **아직 소리 내어 재지 않음**.", head)

out = (head + "\n## 순서와 시간\n" + order_lines + f"\n- 합계 {mmss(total)}. 장별 시간 = 대사 음절 ÷ 초당 3.54음절 + 넘김 약 1.5초(숫자·영문은 가중, 리허설 전 추정).\n\n---\n\n"
       + "\n".join(parts) + "\n---\n\n" + tail)

out = out.replace("- → 왼쪽 두 줄 → 오른쪽 그래프", "- → 왼쪽 세 줄 → 오른쪽 그래프")
# 덱 노트와 대조
said = {}
for blk in re.split(r"\n(?=## \d+\. )", out):
    m = re.match(r"## (\d+)\. [^\n]*\n(.*)", blk, re.S)
    if not m:
        continue
    lines = []
    for ln in m.group(2).split("\n"):
        if ln.startswith("- ") or ln.startswith("---") or ln.startswith("## "):
            break
        if ln.strip():
            lines.append(ln.strip())
    said[int(m.group(1))] = " ".join(lines).replace("**", "")
bad = [i for i in range(1, need_n + 1) if said.get(i) != said_now[i - 1]]
out += f"\n## 대조 기록(집 v6, 2026-11-04)\n- 덱 `팜모니_발표자료_집v6.pptx` 표지 + 본문 15장 노트의 [말할 것]과 이 대본 각 장 대사(`**` 뺀 글자)를 PPTX에서 꺼내 비교: 불일치 {len(bad)}.\n"
open(OUT, "w", encoding="utf-8").write(out)
print("v14 절 재사용", reused, "/", need_n, "합계", mmss(total), "불일치", bad)
print("뒤 절 장 참조 바꾼 곳:")
for c in changes:
    print("  ", c)
