# 대본 v15 → v16: 집 최종 v5 덱(본문 16장)의 노트 [말할 것]으로 장별 대사를 다시 짜고, 질의응답 6문항을 더한다 (2026-11-04, 집 클로드)
# 대사가 v15와 같은 장은 v15 절(굵은 표시·가리킬 곳 포함)을 그대로 쓰고, 바뀐 장은 노트 대사로 새로 쓴다. (변수 이름 v14 = 원본 대본)
import re
import sys

from pptx import Presentation

SRC = "집/클로드/발표/최종/발표대본_집v6_최종_v17_2026-11-04.md"
OUT = "집/클로드/발표/최종/발표대본_집v7_최종_v18_2026-11-04.md"
DECK = "집/클로드/발표/최종/팜모니_발표자료_집v7.pptx"
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
tail = tail[: tail.index("## 대조 기록(집 v6")]
changes = []
tail = tail.replace("그래서 사전 기준 판정 색은 19~22회차에만 칠했습니다.", "사전 기준은 6회차 무렵부터 부품 단위로 판정했고, 그래프는 판정 대신 직전 최고보다 좋아졌는지 나빠졌는지만 보여 줍니다.")
assert "19~22회차에만" not in tail
head = head.replace("# 팜모니 발표 대본 — 최종 v16 (2026-11-04, 집 클로드)",
                    "# 팜모니 발표 대본 — 최종 v17 (2026-11-04, 집 클로드)\n\n"
                    "- v16에서 바뀐 것: 기준 덱 **집 최종 v6**(`집/클로드/발표/최종/팜모니_발표자료_집v6.pptx`, 본문 15장). "
                    "찾은 점을 2장·3개로(③ 추울 때 오차), 1.5 대사 '두 모델을 섞어', 1.6 대사(검증 방식 + 회차 흐름), 질의응답 4문항 추가.")
head = re.sub(r"덱 본문은 1[0-9]장\.", "덱 본문은 15장.", head)
head = head.replace("- 기준 덱: `집/클로드/발표/최종/팜모니_발표자료_집v6.pptx` 본문 1~15장",
                    "- 기준 덱: `집/클로드/발표/최종/팜모니_발표자료_집v7.pptx` 본문 1~15장")
head = re.sub(r"- 길이: [^\n]*", f"- 길이: 음절 기준 추정 합계 {mmss(total)}(넘김 1.5초 포함, 리허설 전). **아직 소리 내어 재지 않음**.", head)

head = head.replace("# 팜모니 발표 대본 — 최종 v17 (2026-11-04, 집 클로드)", "# 팜모니 발표 대본 — 최종 v18 (2026-11-04, 집 클로드)\n\n- v17에서 바뀐 것: 기준 덱 **집 최종 v7**(`집/클로드/발표/최종/팜모니_발표자료_집v7.pptx`). 1.6 회차 그래프를 좋아짐/나빠짐 색으로, 1.6 대사 마지막 두 문장 교체, 질의응답 정정(사전 기준 범위).", 1)
head = head.replace("`집/클로드/발표/최종/팜모니_발표자료_집v6.pptx` 본문 1~15장", "`집/클로드/발표/최종/팜모니_발표자료_집v7.pptx` 본문 1~15장")
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
out += f"\n## 대조 기록(집 v7, 2026-11-04)\n- 덱 `팜모니_발표자료_집v7.pptx` 표지 + 본문 15장 노트의 [말할 것]과 이 대본 각 장 대사(`**` 뺀 글자)를 PPTX에서 꺼내 비교: 불일치 {len(bad)}.\n"
open(OUT, "w", encoding="utf-8").write(out)
print("v14 절 재사용", reused, "/", need_n, "합계", mmss(total), "불일치", bad)
print("뒤 절 장 참조 바꾼 곳:")
for c in changes:
    print("  ", c)
