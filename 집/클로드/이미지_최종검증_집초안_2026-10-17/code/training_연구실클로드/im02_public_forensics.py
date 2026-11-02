"""IM02: 공개 21장(학습 미사용)의 파일 특성 진단 — 배포 파이프라인 추정용.
JPEG 양자화표로 품질 추정, 크기·방향, EXIF/ICC, 서브샘플링, 단순 잡음·선명도 통계."""
import csv, os, numpy as np
from pathlib import Path
from PIL import Image, JpegImagePlugin

PUB = Path(__file__).resolve().parents[3] / "공용/대회자료/이미지데이터/참가자_배포/공개샘플"

# libjpeg 표준 휘도 표(품질 50)
STD_LUMA = np.array([16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,
                     18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99])

def est_quality(q0):
    q0 = np.array(q0)
    best = None
    for Q in range(1, 101):
        s = 5000 / Q if Q < 50 else 200 - 2 * Q
        t = np.clip(np.floor((STD_LUMA * s + 50) / 100), 1, 255)
        # PIL은 지그재그 순서가 아닌 자연 순서로 줄 수 있음 → 정렬 비교로 근사
        err = np.abs(np.sort(t) - np.sort(q0)).sum()
        if best is None or err < best[1]: best = (Q, err)
    return best

def stats(path):
    im = Image.open(path)
    q = im.quantization
    sub = JpegImagePlugin.get_sampling(im)
    g = np.asarray(im.convert("L"), dtype=np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    sat = np.mean((g <= 2) | (g >= 253))
    return dict(size=f"{im.size[0]}x{im.size[1]}", q=est_quality(q[0]), ntab=len(q), sub=sub,
                exif=len(im.getexif()), icc="icc_profile" in im.info, prog=im.info.get("progressive", 0),
                kb=os.path.getsize(path) // 1024, lapstd=float(lap.std()), extreme=float(sat))

rows = list(csv.DictReader(open(PUB / "sample_labels.csv")))
for r in rows:
    s = stats(PUB / "images" / r["file"])
    print(f"{r['file']:9s} {r['type']:13s} {s['size']:9s} Q~{s['q'][0]:3d}(err{s['q'][1]:.0f}) tabs={s['ntab']} sub={s['sub']} "
          f"exif={s['exif']} icc={s['icc']} prog={s['prog']} {s['kb']:4d}KB lapstd={s['lapstd']:6.1f} extreme={s['extreme']:.4f}")
