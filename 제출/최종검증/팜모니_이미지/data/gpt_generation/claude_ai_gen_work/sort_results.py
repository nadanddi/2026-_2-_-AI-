"""앱으로 만든 결과 정리: 결과_<도구>/<유형>/ 안의 아무 이름 파일을 원본 100장과 대조해
<원본>_<유형약자>_<번호>.<확장자>로 이름을 바꾸고 기록.csv에 한 줄씩 적는다. 이미 정리된 이름(src_/g로 시작)은 건너뛴다.
대조: 흑백 24×32 축소 → 표준화 → 상관(구도가 같으면 높음). 최고 상관 < TH면 원본 없음(g번호).
사용: python sort_results.py            (GPT 폴더)
      python sort_results.py --dry      (바꾸지 않고 대조 결과만 출력)
"""
import argparse, csv, re
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
EXT = {".png", ".jpg", ".jpeg", ".webp"}
ABBR = {"1_전체생성": "gen", "2_전체스타일": "fstyle", "3_부분생성": "pgen", "4_부분스타일": "pstyle"}
TH = 0.5

def sig(p):
    a = np.asarray(Image.open(p).convert("L").resize((24, 32), Image.BILINEAR), np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tool", default="GPT"); ap.add_argument("--dry", action="store_true"); g = ap.parse_args()
    srcs = sorted((ROOT / "원본").glob("src_*.jpg")); S = np.stack([sig(p) for p in srcs])
    log = ROOT / "기록.csv"; rows = []
    for typ, ab in ABBR.items():
        d = ROOT / f"결과_{g.tool}" / typ
        if not d.exists(): continue
        used = {f.name for f in d.iterdir()}
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in EXT or re.match(r"^(src_\d{3}|g\d{3})_", f.name): continue
            c = S @ sig(f) / S.shape[1]; i = int(c.argmax()); hit = c[i] >= TH
            stem = srcs[i].stem if hit else "g"
            n = 1
            while True:
                new = f"{stem}_{ab}_{n:03d}{f.suffix.lower()}" if hit else f"g{n:03d}_{ab}{f.suffix.lower()}"
                if new not in used: break
                n += 1
            used.add(new)
            print(f"{typ}/{f.name} → {new}  (상관 {c[i]:.2f}{'' if hit else ', 원본 없음'})")
            if hit and typ == "1_전체생성": print("   ※ 전체생성 폴더인데 원본과 닮음 — 확인 필요")
            if not hit and typ != "1_전체생성": print("   ※ 편집 유형인데 원본을 못 찾음 — 확인 필요(크게 바뀐 결과일 수 있음)")
            rows.append((f, d / new, typ, srcs[i].name if hit else "", round(float(c[i]), 3)))
    if g.dry or not rows: print(f"대상 {len(rows)}개" + (" (dry: 바꾸지 않음)" if g.dry else "")); return
    with open(log, "a", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        for f, new, typ, src, cc in rows:
            f.rename(new); w.writerow([g.tool, typ, new.name, src, f"앱 수동 (원래이름 {f.name}, 대조상관 {cc})"])
    print(f"정리 완료 {len(rows)}개, 기록.csv에 추가")

if __name__ == "__main__":
    main()
