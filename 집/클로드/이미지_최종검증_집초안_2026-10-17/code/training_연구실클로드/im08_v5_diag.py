"""IM08: 팀원 v5(리더보드 .77) 모델을 우리 자료에 돌리는 진단 — 학습 없음, 예측만.
목적: (1) 새 실제 사진 오판율(빨간 조명 포함/제외) (2) 우리 가짜 유형별 탐지율(우리 가짜가 운영진 가짜와 얼마나 닮았는지의 간접 지표)
코드는 압축만 풀어 둔 팀원 inference.py를 경로로 읽음(내용 검토 완료: 네트워크·설치 없음).
사용: python im08_v5_diag.py <set_name> [max_per_kind]
"""
import sys, csv, importlib.util, random, numpy as np
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent; LOCAL = HERE.parent / "local"
SUB = LOCAL / "teammate_img_v5" / "sub"
spec = importlib.util.spec_from_file_location("v5inf", SUB / "inference.py"); v5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v5)

def main(set_name, maxk=150):
    man = list(csv.DictReader(open(LOCAL / set_name / "manifest.csv", encoding="utf-8")))
    red = set((LOCAL / "red_sources.txt").read_text(encoding="utf-8").split(chr(10))) - {""}
    by = defaultdict(list)
    for r in man:
        k = r["kind"] + ("|red" if r["source"] in red else "")
        by[k].append(r)
    rng = random.Random(0); pick = []
    for k, v in by.items(): rng.shuffle(v); pick += [(k, r) for r in v[:maxk]]
    m = v5.load_model(str(SUB / "model"))
    paths = [str(LOCAL / set_name / "img" / r["file"]) for _, r in pick]
    p = np.array(v5.predict(m, paths))
    out = LOCAL / f"v5diag_{set_name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file", "kind", "label", "p"])
        for (k, r), pp in zip(pick, p): w.writerow([r["file"], k, r["label"], f"{pp:.5f}"])
    print(f"[{set_name}] 팀원 v5 예측 (0.5 기준)")
    for k in sorted(by):
        idx = [i for i, (kk, _) in enumerate(pick) if kk == k]
        lab = int(pick[idx[0]][1]["label"]); acc = ((p[idx] >= .5) == lab).mean()
        print(f"  {k:24s} n={len(idx):4d} {'실제 정답률' if lab == 0 else '가짜 탐지율'}={acc:.3f}  평균p={p[idx].mean():.3f}")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 150)
