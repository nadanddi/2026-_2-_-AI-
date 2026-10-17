"""IM32: 봉인 확증 자료 만들기(IM_봉인확증_C3_v1.md). 서버(nadan) 실행.
배포 처리 = 덮게 축소 → 가운데 크롭 768×1024(가로 1024×768) → JPEG Q92 4:2:0 (im03c dist_cover와 같은 규칙, 난수는 파일명 crc32).
출력: local/sealed_c3/<set>/img/*.jpg + manifest.csv (set: real / qwen / v7 / brown / bright)  — 학습용 나머지는 local/im_set_qwen_train, im_set_v7(봉인 제외 목록)으로 별도 사용
"""
import csv, random, zlib, shutil
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im03_make_fakes import jpeg_bytes, LOCAL  # noqa
from im03c_make_fakes_v3 import dist_cover  # noqa
OUT = LOCAL / "sealed_c3"

def put(sn, name, img, row, rows):
    d = OUT / sn / "img"; d.mkdir(parents=True, exist_ok=True); (d / name).write_bytes(jpeg_bytes(img) if img is not None else row.pop("_bytes")); rows.setdefault(sn, []).append(dict(row, file=name))

def main():
    rows = {}
    for r in csv.DictReader(open(LOCAL / "im_new_raw/manifest.csv", encoding="utf-8")):
        if r["role"] in ("sealed", "brown", "bright"):
            sn = {"sealed": "real", "brown": "brown", "bright": "bright"}[r["role"]]
            img = dist_cover(Image.open(LOCAL / "im_new_raw" / r["tar"] / r["file"]).convert("RGB"), random.Random(zlib.crc32(r["file"].encode())))
            put(sn, r["file"], img, dict(label=0, kind="real", group=r["group"], farm_code=r["farm_code"], source=r["file"]), rows)
    for r in csv.DictReader(open(LOCAL / "im_set_qwen/manifest_qwen.csv", encoding="utf-8")):
        if zlib.crc32(r["id"].encode()) % 7 < 3:
            img = dist_cover(Image.open(LOCAL / "im_set_qwen/raw" / f"{r['id']}.png").convert("RGB"), random.Random(zlib.crc32(r["id"].encode())))
            put("qwen", r["id"] + ".jpg", img, dict(label=1, kind="qwen_" + r["kind"], group=r["group"], farm_code=r["farm_code"], source=r["source"]), rows)
    for r in csv.DictReader(open(LOCAL / "im_set_v7/manifest.csv", encoding="utf-8")):
        if zlib.crc32(r["file"].encode()) % 8 < 3:
            put("v7", r["file"], None, dict(label=1, kind="partial_gen_v7", group=r["group"], farm_code=r["farm_code"], source=r["source"], _bytes=(LOCAL / "im_set_v7/img" / r["file"]).read_bytes()), rows)
    for sn, rr in rows.items():
        with open(OUT / sn / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "label", "kind", "group", "farm_code", "source"]); w.writeheader(); w.writerows(rr)
        print(sn, len(rr), flush=True)

if __name__ == "__main__":
    main()
