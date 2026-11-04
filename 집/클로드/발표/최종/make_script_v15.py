# 대본 v14 → v15: 집 최종 v2 덱(본문 15장)의 노트 [말할 것]으로 장별 대사를 다시 짜고, 장 번호 참조를 새 번호로 (2026-11-04, 집 클로드)
# 대사가 v14와 같은 장은 v14 절(굵은 표시·가리킬 곳 포함)을 그대로 쓰고, 바뀐 장은 노트 대사로 새로 쓴다.
import re
import sys

from pptx import Presentation

SRC = "집/클로드/발표/최종/발표대본_집v1_최종_v14_2026-11-04.md"
OUT = "집/클로드/발표/최종/발표대본_집v2_최종_v15_2026-11-04.md"
DECK = "집/클로드/발표/최종/팜모니_발표자료_집v2.pptx"
NBODY = 14

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

# v14의 머리말과 본문 뒤 절(시간 맞추기, 질의응답 …)
head = v14.split("\n## 순서와 시간")[0]
tail = v14[v14.index("## 시간 맞추기"):]
tail = tail[: tail.index("## 대조 기록(G09")]

MAP = {"1.2": "1.1", "1.3": "1.2", "1.4": "1.3", "1.5": "1.4", "1.6": "1.5", "1.8": "1.6", "1.9": "1.7", "2.2": "2.3", "2.3": "2.4", "2.4": "2.5"}
REF = re.compile(r"(?<![\d.vV≤≥=+−~×:\[])([12]\.[2-9])(?=( 노트| 표| 그림|의 |에서|으로|\)| 끄기|\(|·[12]\.|:|,| ·|$))", re.M)
changes = []


def remap(text):
    def f(m):
        if m.group(1) in MAP:
            changes.append(text[max(0, m.start() - 12): m.end() + 6].replace("\n", " "))
            return MAP[m.group(1)]
        return m.group(1)
    return REF.sub(f, text)


tail = remap(tail)
tail = tail.replace('- 1.3: "8도와 10도 사이에서는 비중이 갑자기 바뀌지 않게 했습니다."\n', "")

# 시간 맞추기: 뺄 문장 1·2는 v15에서 이미 뺐음(표·캡션에 있음)
tail = re.sub(r"\*\*9분 30초를 넘으면 차례로 뺄 문장\*\*.*?\n\n",
              "**9분 30초를 넘으면 뺄 문장**(v15에서 화면과 겹치는 대사는 이미 뺐음: 1.6 '표의 오른쪽 열은 …', 1.7 '현장에서는 날짜와 동 ID로 …', '오차는 모두 채점 대상과 같은 기록 번호 구간에서 …')\n"
              "1. 2.5의 \"현장 사진에서도 …\" 한 문장 (약 4초, 화면에 있음)\n"
              "2. 1.2의 \"하루 종일 1도씩 높게 예측했다고 생각해 보세요. 하루 전체가 위로 밀린 겁니다.\" (약 6초)\n\n", tail, count=1, flags=re.S)
tail = tail.replace("어제 오차와 오늘 오차", "전날 오차와 오늘 오차").replace('"어제 오차를 알면', '"전날 오차를 알면')

head = head.replace("# 팜모니 발표 대본 — 최종 v14 (2026-11-04, 집 클로드)",
                    "# 팜모니 발표 대본 — 최종 v15 (2026-11-04, 집 클로드)\n\n"
                    "- v14에서 바뀐 것: 기준 덱 **집 최종 v2**(`집/클로드/발표/최종/팜모니_발표자료_집v2.pptx`, 본문 14장). "
                    "주제마다 앞부분을 '데이터에서 찾은 점 → 모델(근거 표시)', 뒷부분을 운영위 주제로 바꿈. "
                    "정형: 1.1(옛 1.1+1.2), 1.2 데이터에서 찾은 점(옛 1.3 + 그림 1 + ③④), 1.3~1.5 모델, 1.6 검증(옛 1.8), 1.7 해석·현장 적용(옛 1.9+1.7). "
                    "이미지: 2.2 데이터에서 찾은 점(새 장), 2.3~2.5(옛 2.2~2.4). '어제' → '전날'. 대사가 바뀐 장은 굵은 표시와 가리킬 곳 없이 노트 대사 그대로 적음.")
head = re.sub(r"덱 본문은 1[67]장\.", "덱 본문은 14장.", head)
head = head.replace("- 기준 덱: `집/클로드/발표/최종/팜모니_발표자료_집v1.pptx` 본문 1~16장",
                    "- 기준 덱: `집/클로드/발표/최종/팜모니_발표자료_집v2.pptx` 본문 1~14장")
head = re.sub(r"- 길이: [^\n]*", f"- 길이: 음절 기준 추정 합계 {mmss(total)}(넘김 1.5초 포함, 리허설 전). **아직 소리 내어 재지 않음**.", head)

out = (head + "\n## 순서와 시간\n" + order_lines + f"\n- 합계 {mmss(total)}. 장별 시간 = 대사 음절 ÷ 초당 3.54음절 + 넘김 약 1.5초(숫자·영문은 가중, 리허설 전 추정).\n\n---\n\n"
       + "\n".join(parts) + "\n---\n\n" + tail)

out = out.replace("- → 왼쪽 세 줄 → 오른쪽 그래프", "- → 왼쪽 두 줄 → 오른쪽 그래프")
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
out += f"\n## 대조 기록(집 v2, 2026-11-04)\n- 덱 `팜모니_발표자료_집v2.pptx` 표지 + 본문 15장 노트의 [말할 것]과 이 대본 각 장 대사(`**` 뺀 글자)를 PPTX에서 꺼내 비교: 불일치 {len(bad)}.\n"
open(OUT, "w", encoding="utf-8").write(out)
print("v14 절 재사용", reused, "/", need_n, "합계", mmss(total), "불일치", bad)
print("뒤 절 장 참조 바꾼 곳:")
for c in changes:
    print("  ", c)
