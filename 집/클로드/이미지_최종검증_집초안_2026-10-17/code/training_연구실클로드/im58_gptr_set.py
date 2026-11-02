"""IM58: 후보 I'(IM_GPT재그리기_Ip_v1.md) 자료 — 코덱스 GPT 전체 재그리기 PNG → 배포 처리(dist_cover → JPEG Q92) → local/im_set_gptr.
원본 = local/im_i_raw(itrain). 분할 = 촬영 묶음 단위, 큰 묶음부터 장수 적은 쪽에(v2). 짝 진짜(원본 같은 배포 처리) → local/im_set_gptr_src. 서버(nadan). 사용: python im58_gptr_set.py"""
import sys, csv, random, zlib
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im03_make_fakes import jpeg_bytes, LOCAL  # noqa
from im03c_make_fakes_v3 import dist_cover  # noqa
SRC = LOCAL / "gpt_regen_20261015"; OUT = LOCAL / "im_set_gptr"; (OUT / "img").mkdir(parents=True, exist_ok=True)
raw = {r["file"]: r for r in csv.DictReader(open(LOCAL / "im_i_raw/manifest.csv", encoding="utf-8"))}
lst = list(csv.DictReader(open(SRC / "목록.csv", encoding="utf-8-sig"))); rows = []
for r in lst:
    s = raw.get(r["원본파일명"]); p = SRC / r["결과파일명"]
    if s is None or not p.exists(): print("건너뜀", r); continue
    assert s["role"] == "itrain"
    lr = random.Random(zlib.crc32(f"gptr|{r['결과파일명']}".encode())); fn = "gptr_" + Path(r["결과파일명"]).stem + ".jpg"
    (OUT / "img" / fn).write_bytes(jpeg_bytes(dist_cover(Image.open(p).convert("RGB"), lr)))
    rows.append(dict(file=fn, label=1, kind="gpt_regen", info="codex_gpt_regen_20261015", source=s["file"], group=s["group"], farm_code=s["farm_code"], tar=s["tar"], other="",
                     split=""))
import collections   # 분할 v2(결과 전 수정): 묶음 crc 짝/홀이 5묶음 모두 짝이라 hold 0 → 묶음을 이름순으로 장수 반반에 가깝게 배정(큰 묶음부터 적은 쪽에)
cnt = collections.Counter(r["group"] for r in rows); side = {"train": 0, "hold": 0}; asg = {}
for g, n in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
    k = "train" if side["train"] <= side["hold"] else "hold"; asg[g] = k; side[k] += n
for r in rows: r["split"] = asg[r["group"]]
print("묶음 배정", dict(cnt), asg)
with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("GPTR", len(rows), "train", sum(r["split"] == "train" for r in rows), "hold", sum(r["split"] == "hold" for r in rows), "묶음", len({r["group"] for r in rows}))

# 짝 진짜(비평 반영): 같은 원본을 같은 배포 처리 → im_set_gptr_src
O2 = LOCAL / "im_set_gptr_src"; (O2 / "img").mkdir(parents=True, exist_ok=True); r2 = []
for r in rows:
    s_ = raw[r["source"]]; lr = random.Random(zlib.crc32(f"gptr_src|{r['source']}".encode())); fn = "gsrc_" + Path(r["source"]).stem + ".jpg"
    (O2 / "img" / fn).write_bytes(jpeg_bytes(dist_cover(Image.open(LOCAL / "im_i_raw" / s_["tar"] / s_["file"]).convert("RGB"), lr)))
    r2.append(dict(r, file=fn, label=0, kind="real", info="gptr_pair_real"))
with open(O2 / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(r2[0].keys())); w.writeheader(); w.writerows(r2)
print("GPTR_SRC", len(r2))
