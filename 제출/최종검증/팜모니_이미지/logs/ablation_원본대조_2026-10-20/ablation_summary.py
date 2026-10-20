"""preds_ablation.csv → 판(A~D) × 묶음 요약 표(마크다운). 사용: python ablation_summary.py preds_ablation.csv > 요약표.md"""
import sys, csv, collections
TYPE = {"filter_jpeg": "필터", "filter_blur": "필터", "filter_noise": "필터", "filter_sp": "필터", "composite": "합성",
        "partial_gen_v7": "부분 생성", "partial_gen_v7s": "부분 생성", "partial_style": "부분 스타일", "pstyle_color": "부분 스타일",
        "regen_xl": "전체 스타일", "partial_gen": "부분 생성", "partial_style_pub": "부분 스타일", "AI_gen": "전체 생성", "full_style": "전체 스타일", "filter": "필터"}
TRAIN = {"im_set_gpt1": "표본 안", "im_set_gpt2": "표본 안", "im_set_gpt3": "표본 안", "im_set_gptr": "표본 안"}
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
by = collections.defaultdict(list)
for r in rows: by[r["set"]].append(r)
V = "ABCD"; NAME = {"A": "A 그대로", "B": "B R1 끔", "C": "C 정밀 대조 끔", "D": "D 지문 DB 전부 끔"}
def tp(r):
    if r["set"].startswith("im_set_gpt"): return "GPT:" + r["kind"]
    return TYPE.get(r["kind"], r["kind"])
print("## 1. 판 × 묶음 (0.5 기준)\n")
print("| 묶음 | 학습 사용 | 장수(실제/가짜) | 판 | 균형 정확도 | 실제 오탐률 | 가짜 놓침률 | A 대비 실제→가짜 | A 대비 가짜→실제 |")
print("|---|---|---|---|---|---|---|---|---|")
for sn, rr in by.items():
    y = [int(r["label"]) for r in rr]; nr = y.count(0); nf = y.count(1)
    pa = [float(r["pA"]) >= .5 for r in rr]
    for v in V:
        pv = [float(r["p" + v]) >= .5 for r in rr]
        fp = sum(p and t == 0 for p, t in zip(pv, y)) / nr if nr else None
        fn = sum((not p) and t == 1 for p, t in zip(pv, y)) / nf if nf else None
        ba = .5 * (1 - fp) + .5 * (1 - fn) if nr and nf else None
        r2f = sum(p and not a and t == 0 for p, a, t in zip(pv, pa, y)); f2r = sum((not p) and a and t == 1 for p, a, t in zip(pv, pa, y))
        f = lambda x: "—" if x is None else f"{x:.4f}"
        print(f"| {sn} | {TRAIN.get(sn, '미사용')} | {nr}/{nf} | {NAME[v]} | {f(ba)} | {f(fp)} | {f(fn)} | {'—' if v == 'A' else r2f} | {'—' if v == 'A' else f2r} |")
print("\n## 2. 가짜 유형별 탐지율 (묶음 합산)\n")
grp = collections.OrderedDict((("봉인 R2~R5", ("im_set_r2", "im_set_r3", "im_set_r4", "im_set_r5")), ("공개 21", ("pub21",)), ("GPT 4묶음(표본 안)", ("im_set_gpt1", "im_set_gpt2", "im_set_gpt3", "im_set_gptr"))))
for g, ss in grp.items():
    c = collections.defaultdict(lambda: [0] * 5)
    for sn in ss:
        for r in by.get(sn, []):
            if r["label"] != "1": continue
            k = tp(r); c[k][0] += 1
            for j, v in enumerate(V): c[k][j + 1] += float(r["p" + v]) >= .5
    print(f"\n### {g}\n\n| 유형 | 장수 | A | B | C | D |\n|---|---|---|---|---|---|")
    for k in sorted(c): n = c[k][0]; print(f"| {k} | {n} | " + " | ".join(f"{c[k][j] / n:.3f}" for j in range(1, 5)) + " |")
