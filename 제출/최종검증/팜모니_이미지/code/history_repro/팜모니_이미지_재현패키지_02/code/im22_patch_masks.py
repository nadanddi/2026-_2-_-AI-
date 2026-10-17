"""IM22: 후보 L(IM_국소패치_L_v1.md) 조각 격자와 마스크 겹침.
- grid(w,h): 224 창, 112 간격 → 54조각 (가로·세로 모두).
- 서버: 6판 부분생성·5판 partial_style_xl 마스크를 생성 시드로 재현(파이프라인 호출 없이 같은 난수 순서) → 기록된 area와 검산 → local/feats/<set>_pmask.npz (files, ov=(N,54))
사용: python im22_patch_masks.py
"""
import sys, csv, re, random, zlib, numpy as np
from pathlib import Path
from PIL import Image
S, ST = 224, 112

def grid(w, h):
    xs = np.linspace(0, w - S, int(np.ceil((w - S) / ST)) + 1).round().astype(int)
    ys = np.linspace(0, h - S, int(np.ceil((h - S) / ST)) + 1).round().astype(int)
    return [(int(x), int(y)) for y in ys for x in xs]

def overlap(m):
    h, w = m.shape; return np.array([m[y:y + S, x:x + S].mean() for x, y in grid(w, h)], np.float32)

def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from im03_make_fakes import to_np, blob_mask, LOCAL  # noqa
    from im03c_make_fakes_v3 import dist_cover  # noqa
    from im15_fakes_v5 import rawp, sources, object_mask, PEDIT, ART  # noqa
    man = sources()
    jobs = {"im_set_v6": [r for r in csv.DictReader(open(LOCAL / "im_set_v6/manifest.csv", encoding="utf-8"))],
            "im_set_v5": [r for r in csv.DictReader(open(LOCAL / "im_set_v5/manifest.csv", encoding="utf-8")) if r["kind"] == "partial_style_xl"]}
    for sn, rows in jobs.items():
        files, ovs, bad = [], [], 0
        for r in rows:
            i = int(re.search(r"_(\d{5})\.jpg", r["file"]).group(1))
            if sn == "im_set_v6":
                lr = random.Random(zlib.crc32(f"v6|pgen|{i}".encode())); lr.randint(0, 2**31); src = man[lr.randrange(len(man))]
                a = to_np(dist_cover(Image.open(rawp(src["file"])).convert("RGB"), lr))
                m = object_mask(a, lr, .02, .15) if lr.random() < .5 else blob_mask(a.shape, lr, (.02, .15))
            else:
                lr = random.Random(zlib.crc32(f"v5|partial_style_xl|{i}".encode())); lr.randint(0, 2**31); src = man[lr.randrange(len(man))]
                a = to_np(dist_cover(Image.open(rawp(src["file"])).convert("RGB"), lr)); lr.randint(25, 35); lr.uniform(5, 8)
                photo = lr.random() < .5; lr.uniform(.30, .45) if photo else lr.uniform(.40, .55); lr.choice(PEDIT if photo else ART)
                m = object_mask(a, lr)
            want = float(re.search(r"area=([\d.]+)", r["info"]).group(1)); ok = src["file"] == r["source"] and abs(m.mean() - want) < .006
            if not ok: bad += 1; continue
            files.append(r["file"]); ovs.append(overlap(m.astype(np.float32)))
        np.savez(LOCAL / "feats" / f"{sn}_pmask.npz", files=np.array(files), ov=np.stack(ovs))
        print(sn, "재현", len(files), "불일치", bad, "양성조각(≥.25) 평균", float((np.stack(ovs) >= .25).sum(1).mean()), flush=True)

if __name__ == "__main__":
    main()
