"""IM21c: 코덱스 GPT v3(더 강한 편집) → local/im_set_gpt3 (IM_GPT학습_G3_v1.md 시험대). im21과 같은 변환.
최종(3_부분생성·4_부분스타일)과 1단계(중간_GPT)를 모두 넣는다. 원본 번호 → 원본3_목록.csv로 AI허브 원천·농장.
편집 종류는 manifest_v3.csv가 있으면 그 '편집'/'action' 열을 info에 기록.
"""
import csv, random, zlib
from pathlib import Path
from collections import Counter
from PIL import Image
from im03_make_fakes import jpeg_bytes, LOCAL  # noqa
from im03c_make_fakes_v3 import dist_cover  # noqa
SRC = Path(__file__).resolve().parents[2] / "코덱스/local/ai_gen_work_20261014_v3"; OUT = LOCAL / "im_set_gpt3"
SUB = [("결과_GPT/3_부분생성", "gpt3_pgen"), ("결과_GPT/4_부분스타일", "gpt3_pstyle"), ("중간_GPT", "gpt3_s1")]

def main():
    lst = {r["파일명"].split(".")[0]: r for r in csv.DictReader(open(LOCAL / "ai_gen_work/원본3_목록.csv", encoding="utf-8-sig"))}
    act = {}
    mp = SRC / "manifest_v3.csv"
    if mp.exists():
        for r in csv.DictReader(open(mp, encoding="utf-8-sig")):
            k = Path(r.get("file", "")).name
            act[k] = "|".join(str(r.get(c, "")) for c in r if any(s in c for s in ("방식", "편집", "action", "종류")))
    (OUT / "img").mkdir(parents=True, exist_ok=True); rows = []
    for sub, kind in SUB:
        for p in sorted((SRC / sub).glob("*.png")):
            sid = "_".join(p.name.split("_")[:2]); src = lst[sid]["AI허브_원천"]; stem = Path(src.replace("\\", "/")).stem
            name = f"{kind}_{len(rows):03d}.jpg"; img = dist_cover(Image.open(p).convert("RGB"), random.Random(zlib.crc32(p.name.encode())))
            (OUT / "img" / name).write_bytes(jpeg_bytes(img))
            rows.append(dict(file=name, label=1, kind=kind, info=f"gpt_v3;{p.name};{act.get(p.name, '')}", source=src, group="_".join(stem.split("_")[:3]),
                             farm_code=stem.split("_")[0], tar=src.replace("\\", "/").split("/")[0], other=sid + ".jpg"))
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(rows), Counter(r["kind"] for r in rows), len({r["farm_code"] for r in rows}), "농장", "편집기록", sum(1 for r in rows if r["info"].split(";")[2]))

if __name__ == "__main__":
    main()
