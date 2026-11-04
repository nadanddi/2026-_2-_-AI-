# 대본 v15 → v16: 집 최종 v5 덱(본문 16장)의 노트 [말할 것]으로 장별 대사를 다시 짜고, 질의응답 6문항을 더한다 (2026-11-04, 집 클로드)
# 대사가 v15와 같은 장은 v15 절(굵은 표시·가리킬 곳 포함)을 그대로 쓰고, 바뀐 장은 노트 대사로 새로 쓴다. (변수 이름 v14 = 원본 대본)
import re
import sys

from pptx import Presentation

SRC = "집/클로드/발표/최종/발표대본_집v2_최종_v15_2026-11-04.md"
OUT = "집/클로드/발표/최종/발표대본_집v5_최종_v16_2026-11-04.md"
DECK = "집/클로드/발표/최종/팜모니_발표자료_집v5.pptx"
NBODY = 16

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
tail = tail[: tail.index("## 대조 기록(집 v2")]
changes = []
tail = tail.replace("2. 1.2의 \"하루 종일 1도씩 높게 예측했다고 생각해 보세요. 하루 전체가 위로 밀린 겁니다.\" (약 6초)",
                    "2. 1.2 (1/3)의 \"오른쪽 실제 하루처럼, 예측이 하루 내내 거의 같은 간격으로 낮습니다.\" (약 5초, 그림으로 보임)")
QA16 = """
- **(집 v5) 그림 5 반영률이 왜 9시 이후 다시 오르나? 23시에도 20% 넘게 반영하나?** 하루 단위로 밀리는 오차 때문에 저녁에도 조금은 남는 게 맞습니다(큰 표본 318개에서 23시 약 8%). 24%는 확신 높은 학습 기록 32개로 시각마다 따로 잰 흔들림이 그대로 들어간 값이고, 9시 이후 상승은 통계적으로 확인되지 않았습니다(구간 [−0.12, +0.22]). 줄어드는 곡선으로 다시 맞추는 것이 다음 할 일입니다. · 1.4 노트
- **(집 v5) A·B는 어떤 모델인가?** 배지가 실내 온도·일사·난방·외기를 몇 시간~며칠 늦게 따라가는 성질을 선형 회귀로 잡고, 남은 오차를 부스팅 트리(LightGBM)로 고칩니다. · 1.3 노트
- **(집 v5) 사전학습 트랜스포머(TabPFN)는 무엇인가?** 합성 표 데이터로 미리 학습된 공개 모델이고, 우리 자료로 재학습하지 않고 학습 기록을 예시로 보여 주고 예측합니다. · 1.3 노트
- **(집 v5) EC 기본 예측은?** 트리 두 종류와 작은 신경망을 섞고(ExtraTrees 60·LightGBM 30·MLP 10), 사전학습 트랜스포머를 20% 더합니다. · 1.5 노트
- **(집 v5) RNN은 안 썼나?** 개발 중 GRU를 시험했지만 최종 구성에는 없습니다.
- **(집 v5) 예시 하루(그림 1)는 어떻게 골랐나?** 오차가 큰 상위 10% 날 중 하루 밀림이 가장 뚜렷한 날을 미리 정한 규칙으로 골랐습니다. 모든 날이 이렇지는 않고, 표의 숫자가 전체 평균입니다. · 1.2 (1/3) 노트
"""
assert tail.count("\n### 나. 답이 약한 질문") == 1
tail = tail.replace("\n### 나. 답이 약한 질문", QA16 + "\n### 나. 답이 약한 질문", 1)

head = head.replace("# 팜모니 발표 대본 — 최종 v15 (2026-11-04, 집 클로드)",
                    "# 팜모니 발표 대본 — 최종 v16 (2026-11-04, 집 클로드)\n\n"
                    "- v15에서 바뀐 것: 기준 덱 **집 최종 v5**(`집/클로드/발표/최종/팜모니_발표자료_집v5.pptx`, 본문 16장). "
                    "1.2를 세 장으로(① 표 + 실제 예시 하루 F47_189 그래프 / ② 그림 / ③④ 막대그래프), 1.3 대사의 물리 문장 삭제, 1.4 대사 한 문장 교체, "
                    "1.5에 '온도와 달리 EC는 전날 실측을 하루 내내 같은 비율로 반영' 한 문장, 질의응답 6문항 추가.")
head = re.sub(r"덱 본문은 1[0-9]장\.", "덱 본문은 16장.", head)
head = head.replace("- 기준 덱: `집/클로드/발표/최종/팜모니_발표자료_집v2.pptx` 본문 1~14장",
                    "- 기준 덱: `집/클로드/발표/최종/팜모니_발표자료_집v5.pptx` 본문 1~16장")
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
out += f"\n## 대조 기록(집 v5, 2026-11-04)\n- 덱 `팜모니_발표자료_집v5.pptx` 표지 + 본문 16장 노트의 [말할 것]과 이 대본 각 장 대사(`**` 뺀 글자)를 PPTX에서 꺼내 비교: 불일치 {len(bad)}.\n"
open(OUT, "w", encoding="utf-8").write(out)
print("v14 절 재사용", reused, "/", need_n, "합계", mmss(total), "불일치", bad)
print("뒤 절 장 참조 바꾼 곳:")
for c in changes:
    print("  ", c)
