"""IM21b: 코덱스 GPT 부분 편집 추가 150장(v2) → local/im_set_gpt2 (IM_부분생성_D1c_v1.md 시험대). im21과 같은 변환."""
import csv, random, zlib
from pathlib import Path
from collections import Counter
from PIL import Image
from im03_make_fakes import jpeg_bytes, LOCAL  # noqa
from im03c_make_fakes_v3 import dist_cover  # noqa
from im21_gpt_set import KIND  # noqa

SRC = Path(__file__).resolve().parents[2] / "코덱스/local/ai_gen_work_20261013_v2"
OUT = LOCAL / "im_set_gpt2"

def main():
    (OUT / "img").mkdir(parents=True, exist_ok=True); rows = []
    for r in csv.DictReader(open(SRC / "manifest_v2.csv", encoding="utf-8-sig")):
        kind = KIND[r["type"]]; src = r["aihub_source"]; name = f"{kind}2_{len(rows):03d}.jpg"
        img = dist_cover(Image.open(SRC / r["file"]).convert("RGB"), random.Random(zlib.crc32(r["file"].encode())))
        (OUT / "img" / name).write_bytes(jpeg_bytes(img)); stem = Path(src.replace("\\", "/")).stem
        rows.append(dict(file=name, label=1, kind=kind, info=f"gpt_imagegen_v2;{r['file']};{img.size[0]}x{img.size[1]}", source=src,
                         group="_".join(stem.split("_")[:3]), farm_code=stem.split("_")[0], tar=src.replace("\\", "/").split("/")[0], other=r["source"]))
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(rows), Counter(r["kind"] for r in rows), len({r["farm_code"] for r in rows}), "농장")

if __name__ == "__main__":
    main()
