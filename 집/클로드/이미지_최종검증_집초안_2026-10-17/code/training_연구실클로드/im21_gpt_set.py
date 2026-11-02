"""IM21: 코덱스 GPT 생성·편집 100장 → 배포 형식 학습 세트 local/im_set_gpt1 (IM_GPT가짜_v1.md).
원본 PNG(코덱스 폴더)는 읽기만 한다. 덮게 축소 → 가운데 크롭 → JPEG Q92 4:2:0 (im03c dist_cover와 같은 규칙).
"""
import csv, random, zlib
from pathlib import Path
from PIL import Image
from im03_make_fakes import jpeg_bytes, LOCAL  # noqa
from im03c_make_fakes_v3 import dist_cover  # noqa

SRC = Path(__file__).resolve().parents[2] / "코덱스/local/ai_gen_work_20261013_v1"
OUT = LOCAL / "im_set_gpt1"
KIND = {"1_전체생성": "gpt_gen", "2_전체스타일": "gpt_fstyle", "3_부분생성": "gpt_pgen", "4_부분스타일": "gpt_pstyle"}

def main():
    (OUT / "img").mkdir(parents=True, exist_ok=True); rows = []
    for r in csv.DictReader(open(SRC / "manifest_v1.csv", encoding="utf-8-sig")):
        kind = KIND[r["type"]]; src = r["aihub_source"] or ""
        name = f"{kind}_{len(rows):03d}.jpg"; rng = random.Random(zlib.crc32(r["file"].encode()))
        img = dist_cover(Image.open(SRC / r["file"]).convert("RGB"), rng)
        (OUT / "img" / name).write_bytes(jpeg_bytes(img))
        stem = Path(src.replace("\\", "/")).stem; farm = stem.split("_")[0] if src else "GPTGEN"
        rows.append(dict(file=name, label=1, kind=kind, info=f"gpt_imagegen;{r['file']};{img.size[0]}x{img.size[1]}", source=src,
                         group="_".join(stem.split("_")[:3]) if src else f"GPTGEN_{len(rows):03d}", farm_code=farm,
                         tar=src.replace("\\", "/").split("/")[0] if src else "", other=r["source"]))
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(len(rows), Counter(r["kind"] for r in rows), Counter(r["farm_code"] for r in rows))

if __name__ == "__main__":
    main()
