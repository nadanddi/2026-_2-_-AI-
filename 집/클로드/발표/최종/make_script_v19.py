# 대본 v18 → v19: 사용자가 직접 고쳐 제출한 덱(제출본, 2026-11-04)의 노트 [말할 것]에 대사를 맞춘다 (2026-11-05, 집 클로드)
# 대사가 v18과 같은 장은 v18 절(굵은 표시·가리킬 곳 포함)을 그대로 쓰고, 바뀐 장은 노트 대사로 새로 쓴다.
import re

from pptx import Presentation

SRC = "집/클로드/발표/최종/발표대본_집v7_최종_v18_2026-11-04.md"
OUT = "집/클로드/발표/최종/발표대본_제출본_최종_v19_2026-11-05.md"
DECK = "집/클로드/발표/최종/팜모니_발표자료_제출본_2026-11-04.pptx"
NBODY = 15

v18 = open(SRC, encoding="utf-8").read()

sec = {}
for blk in re.split(r"\n(?=## \d+\. )", v18):
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
need_n = NBODY + 1


def secs(t):
    syl = len(re.findall(r"[가-힣]", t)) + len(re.findall(r"[0-9]", t)) * 0.8 + len(re.findall(r"[A-Za-z]", t)) * 0.4
    return syl / 3.54 + 1.5


def mmss(x):
    x = round(x)
    return f"{x // 60}:{x % 60:02d}"


parts, times, changed = [], [], []
for i, (t, s_) in enumerate(zip(tags, said_now), 1):
    tm = secs(s_)
    times.append(tm)
    if s_ in sec:
        body = sec[s_]
    else:
        body = s_
        changed.append(i)
    parts.append(f"## {i}. {t} ({mmss(tm)})\n{body}\n")

total = sum(times)
order_lines = "\n".join(f"- {tags[i]} {mmss(times[i])}" for i in range(len(tags)))

head = v18.split("\n## 순서와 시간")[0]
tail = v18[v18.index("## 시간 맞추기"):]
tail = tail[: tail.index("## 대조 기록(집 v7")]
DECKREF = "`집/클로드/발표/최종/팜모니_발표자료_제출본_2026-11-04.pptx`"
head = head.replace("# 팜모니 발표 대본 — 최종 v18 (2026-11-04, 집 클로드)",
                    "# 팜모니 발표 대본 — 최종 v19 (2026-11-05, 집 클로드)\n\n"
                    f"- v18에서 바뀐 것: 기준 덱 **제출본**({DECKREF}, 사용자가 집 v7을 직접 고쳐 11-04 제출한 파일). "
                    f"노트 대사가 바뀐 장({', '.join(str(c) for c in changed)}번 절)을 제출본 대사로 교체. "
                    "제출본 2.4 화면에서 '검증 묶음은 후보 선택에 씀 · 새 농장 성능 미검증' 줄이 빠졌으므로, 2.4 대사의 이 부분은 **반드시 말로** 할 것.", 1)
head = head.replace("`집/클로드/발표/최종/팜모니_발표자료_집v7.pptx` 본문 1~15장", f"{DECKREF} 본문 1~15장")
head = re.sub(r"- 길이: [^\n]*", f"- 길이: 음절 기준 추정 합계 {mmss(total)}(넘김 1.5초 포함, 리허설 전). **아직 소리 내어 재지 않음**.", head)
OLDPT = "- → 표 위에서 아래로 → 표 아래 운영진 한 줄 → 아래 두 줄 → 맨 아래 개선 사례"
assert sum(b.count(OLDPT) for b in parts) == 1
parts = [b.replace(OLDPT, "- → 표 위에서 아래로 → '다만 …' 두 문장은 화면에 없으니(제출본에서 줄 삭제) 청중을 보며 말로 강조") for b in parts]
out = (head +"\n## 순서와 시간\n" + order_lines + f"\n- 합계 {mmss(total)}. 장별 시간 = 대사 음절 ÷ 초당 3.54음절 + 넘김 약 1.5초(숫자·영문은 가중, 리허설 전 추정).\n\n---\n\n"
       + "\n".join(parts) + "\n---\n\n" + tail)

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
out += f"\n## 대조 기록(제출본, 2026-11-05)\n- 덱 `팜모니_발표자료_제출본_2026-11-04.pptx` 표지 + 본문 15장 노트의 [말할 것]과 이 대본 각 장 대사(`**` 뺀 글자)를 PPTX에서 꺼내 비교: 불일치 {len(bad)}.\n"
open(OUT, "w", encoding="utf-8").write(out)
print("바뀐 절", changed, "합계", mmss(total), "불일치", bad)
for c in changed:
    print(c, tags[c - 1], "|", said_now[c - 1])
