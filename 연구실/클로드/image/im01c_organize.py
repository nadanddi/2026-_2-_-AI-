"""IM01c: 추출한 실제 원본 사진을 품종별 폴더로 정리 (사용자 지시 10-10). 하드링크라 디스크 추가 사용 없음(실패 시 복사).
local/im_real_sorted/
  금실/            <tar>__<파일명>.jpg
  설향/
  설향_빨간조명_제외/   (사용자 결정으로 학습 제외)
  목록.csv          폴더·원본 경로·tar·개체 그룹·코드
  요약.txt          품종별 장수·비중
"""
import csv, os, shutil
from pathlib import Path
from collections import Counter

LOCAL = Path(__file__).resolve().parents[1] / "local"; RAW = LOCAL / "im_real_raw"; OUT = LOCAL / "im_real_sorted"

def main():
    man = list(csv.DictReader(open(RAW / "manifest.csv", encoding="utf-8")))
    red = set((LOCAL / "red_sources.txt").read_text(encoding="utf-8").split(chr(10))) - {""}
    rows = []; c = Counter()
    for r in man:
        v = "금실" if "금실" in r["tar"] else "설향"
        folder = "설향_빨간조명_제외" if r["file"] in red else v
        d = OUT / folder; d.mkdir(parents=True, exist_ok=True)
        dst = d / f"{r['tar']}__{Path(r['file']).name}"
        if not dst.exists():
            try: os.link(RAW / r["file"], dst)
            except OSError: shutil.copy2(RAW / r["file"], dst)
        rows.append(dict(folder=folder, file=dst.name, source=r["file"], tar=r["tar"], group=r["group"], code=r["farm_code"])); c[folder] += 1
    with open(OUT / "목록.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    used = c["금실"] + c["설향"]
    txt = [f"실제 원본 정리 (학습 사용 = 금실 + 설향, 빨간 조명 제외)",
           f"금실: {c['금실']}장 ({c['금실'] / used:.1%})", f"설향: {c['설향']}장 ({c['설향'] / used:.1%})",
           f"설향_빨간조명_제외: {c['설향_빨간조명_제외']}장 (학습 미사용)", f"합계 파일: {sum(c.values())}"]
    (OUT / "요약.txt").write_text("\n".join(txt), encoding="utf-8"); print("\n".join(txt))

if __name__ == "__main__":
    main()
