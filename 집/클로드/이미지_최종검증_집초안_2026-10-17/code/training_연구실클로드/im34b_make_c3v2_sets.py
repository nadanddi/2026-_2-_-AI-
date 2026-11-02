"""IM34b: C3v2 학습 묶음 만들기 v2(IM_C3v2_v2.md) — 갈변·밝음 묶음 단위 분할, 봉인 Qwen·7판 원본을 쓴 사진은 학습 제외. 서버(nadan).
 im_set_newreal: 새 AI Hub train 600 + brown/bright 중 crc32%2==0 절반 (배포 처리)
 im_set_v7tr   : 7판 중 봉인 제외(crc32%8>=3) — 이미 배포 형식 jpg 복사 + 마스크 겹침 pmask.npz
 im_set_qwentr : Qwen 중 봉인 제외(crc32%7>=3), 배포 처리. kind: qwen_pgen/qwen_pstyle/qwen_pcolor/qwen_brown/qwen_regen/qwen_fstyle
판정용으로 남는 갈변·밝음: 촬영 묶음 crc32%2==1 (채점 시 필터)
"""
import csv, random, zlib, shutil, sys
import numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im03_make_fakes import jpeg_bytes, LOCAL  # noqa
from im03c_make_fakes_v3 import dist_cover  # noqa
from im22_patch_masks import overlap  # noqa
FIELDS = ["file", "label", "kind", "info", "source", "group", "farm_code", "tar", "other"]

def write(sn, rows):
    with open(LOCAL / sn / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(sn, len(rows), flush=True)

def main():
    sealed_src = set()
    for r in csv.DictReader(open(LOCAL / "im_set_qwen/manifest_qwen.csv", encoding="utf-8")):
        if zlib.crc32(r["id"].encode()) % 7 < 3: sealed_src.add(r["source"])
    for r in csv.DictReader(open(LOCAL / "im_set_v7/manifest.csv", encoding="utf-8")):
        if zlib.crc32(r["file"].encode()) % 8 < 3: sealed_src.add(r["source"])
    print("봉인 가짜 원본", len(sealed_src), flush=True)
    rows = []; (LOCAL / "im_set_newreal/img").mkdir(parents=True, exist_ok=True)
    for r in csv.DictReader(open(LOCAL / "im_new_raw/manifest.csv", encoding="utf-8")):
        if r["role"] == "train" or (r["role"] in ("brown", "bright") and zlib.crc32(r["group"].encode()) % 2 == 0):
            img = dist_cover(Image.open(LOCAL / "im_new_raw" / r["tar"] / r["file"]).convert("RGB"), random.Random(zlib.crc32(r["file"].encode())))
            (LOCAL / "im_set_newreal/img" / r["file"]).write_bytes(jpeg_bytes(img))
            rows.append(dict(file=r["file"], label=0, kind="real", info=f"newreal;{r['role']}", source=r["file"], group=r["group"], farm_code=r["farm_code"], tar=r["tar"], other=""))
    write("im_set_newreal", rows)
    rows, files, ovs, areas = [], [], [], []; (LOCAL / "im_set_v7tr/img").mkdir(parents=True, exist_ok=True)
    for r in csv.DictReader(open(LOCAL / "im_set_v7/manifest.csv", encoding="utf-8")):
        if zlib.crc32(r["file"].encode()) % 8 >= 3 and r["source"] not in sealed_src:
            shutil.copyfile(LOCAL / "im_set_v7/img" / r["file"], LOCAL / "im_set_v7tr/img" / r["file"]); rows.append(dict(r))
            m = (np.asarray(Image.open(LOCAL / "im_set_v7/mask" / r["file"].replace(".jpg", ".png"))) > 127).astype(np.float32)
            files.append(r["file"]); ovs.append(overlap(m)); areas.append(float(m.mean()))
    write("im_set_v7tr", rows)
    np.savez(LOCAL / "feats/im_set_v7tr_pmask.npz", files=np.array(files), ov=np.stack(ovs), area=np.array(areas), ok=np.ones(len(files), bool))
    rows = []; (LOCAL / "im_set_qwentr/img").mkdir(parents=True, exist_ok=True)
    km = {"fstyle_photo": "fstyle"}
    for r in csv.DictReader(open(LOCAL / "im_set_qwen/manifest_qwen.csv", encoding="utf-8")):
        if zlib.crc32(r["id"].encode()) % 7 >= 3 and r["source"] not in sealed_src:
            img = dist_cover(Image.open(LOCAL / "im_set_qwen/raw" / f"{r['id']}.png").convert("RGB"), random.Random(zlib.crc32(r["id"].encode())))
            fn = r["id"] + ".jpg"; (LOCAL / "im_set_qwentr/img" / fn).write_bytes(jpeg_bytes(img))
            rows.append(dict(file=fn, label=1, kind="qwen_" + km.get(r["kind"], r["kind"]), info="qwen_edit;" + r["prompt"][:80].replace(",", " "), source=r["source"], group=r["group"], farm_code=r["farm_code"], tar=r["tar"], other=r["id"]))
    write("im_set_qwentr", rows)

if __name__ == "__main__":
    main()
