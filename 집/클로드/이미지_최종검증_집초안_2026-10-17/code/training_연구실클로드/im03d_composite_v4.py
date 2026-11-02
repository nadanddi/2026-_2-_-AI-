"""IM03d: 가짜 v4 = v3에서 합성(composite)만 공식 정의에 맞춰 다시 만든 세트 (사용자 지시 10-11, '2번').
공식: "작물 영역 등 이미지의 일부를 다른 배경·영역과 합친 이미지". 공개 예시(bpem·bphr·yuag) = 작물을 분할해 다른 배경에 놓은 형태.
v3 합성 탐지율: 배경교체 .88, 여백 .83, 잎조각 붙여넣기 .24~.30 → 붙여넣기 비중 축소, '작물 전체 분할 → 다른 장면 배경' 위주.
모드(확률): plant_on_scene .40 (다른 실제 사진 = 다른 개체/품종 배경 위에 작물 전체)
            plant_on_blur  .25 (흐리게·탈색한 배경 위에 작물 전체, bpem형)
            plant_on_plain .10 (단색/그라데이션 배경)
            margin         .10 (흰 여백, yuag형)
            paste          .15 (v3 잎조각 붙여넣기 — 소수 유지)
마스크: 녹색(HSV) + 꽃(흰)·열매(빨강) 영역, 형태학 정리 후 큰 연결요소. 경계 페더 0~4px, 작물 쪽 색·밝기 미세 보정(50%).
v3의 다른 행(실제·필터·스타일)은 하드링크로 그대로 가져와 같은 원본·같은 파일.
"""
import csv, os, random, zlib, shutil, sys
import numpy as np, cv2
from pathlib import Path
from PIL import Image
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im03_make_fakes import jpeg_bytes, to_np, to_img, feather, RAW, LOCAL  # noqa
import im03b_make_fakes_v2 as v2  # noqa
from im03c_make_fakes_v3 import dist_cover  # noqa

SRC = LOCAL / "im_set_v3"; OUT = LOCAL / "im_set_v4"

def crop_mask(a):
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV); H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (H >= 25) & (H <= 95) & (S >= 45) & (V >= 30)
    flower = (S <= 40) & (V >= 200)                           # 흰 꽃(흰 시설과 섞일 수 있어 녹색 근처만)
    fruit = ((H <= 8) | (H >= 170)) & (S >= 90) & (V >= 60)
    g = green.astype(np.uint8); near = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    m = (green | (fruit & (near > 0))).astype(np.uint8)   # 흰 꽃 항목은 흰 홈통까지 잡아 제외(10-11 육안 확인)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k, iterations=2)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > .004 * m.size: keep[lab == i] = 1
    # 내부 구멍 메우기
    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(keep); cv2.drawContours(filled, cnts, -1, 1, -1)
    return np.where((filled > 0) & (m > 0) | (keep > 0), 1, 0).astype(np.uint8)

def adjust(plant, rng):
    if rng.random() < .5: return plant
    o = plant.astype(np.float32) * rng.uniform(.9, 1.12) + rng.uniform(-10, 10)
    return np.clip(o, 0, 255)

def compose(a, rng, other):
    h, w = a.shape[:2]
    mode = rng.choices(["plant_on_scene", "plant_on_blur", "plant_on_plain", "margin", "paste"], [.40, .25, .10, .10, .15])[0]
    if mode in ("margin", "paste"):
        o, _ = v2.make_fake(a, "composite", _Forced(rng, mode), other); return o, mode
    m = crop_mask(a)
    if m.mean() < .05: mode = "plant_on_blur"
    if mode == "plant_on_scene":
        b = other.astype(np.float32)
        if rng.random() < .5:   # 배경 쪽 작물을 줄여 '다른 장면' 느낌(살짝 흐림)
            b = cv2.GaussianBlur(b, (0, 0), rng.uniform(.5, 2))
    elif mode == "plant_on_blur":
        src = other if rng.random() < .5 else a
        b = cv2.GaussianBlur(src, (0, 0), rng.uniform(2, 8)).astype(np.float32)
        if rng.random() < .6:
            gb = cv2.cvtColor(cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB).astype(np.float32)
            b = b * .4 + gb * .6
    else:
        c1, c2 = np.array([rng.randint(60, 255) for _ in range(3)]), np.array([rng.randint(60, 255) for _ in range(3)])
        tt = np.linspace(0, 1, h)[:, None, None]; b = (c1 * (1 - tt) + c2 * tt) * np.ones((h, w, 3))
    if rng.random() < .15:   # 작물 위치·크기 이동(잘린 직선 경계가 생겨 비중 축소)
        s = rng.uniform(.75, 1.0); M = np.float32([[s, 0, rng.uniform(0, (1 - s) * w)], [0, s, rng.uniform(0, (1 - s) * h)]])
        a2 = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_AREA); m = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST)
    else: a2 = a
    f = feather(m, rng.randint(0, 4))[..., None]
    return np.clip(adjust(a2, rng) * f + b * (1 - f), 0, 255).astype(np.uint8), mode

class _Forced(random.Random):
    """v2.make_fake의 composite 모드 선택을 고정하기 위한 래퍼"""
    def __init__(self, rng, mode): super().__init__(rng.random()); self._mode = mode
    def choices(self, pop, weights=None, k=1):
        target = {"margin": "margin", "paste": "paste"}[self._mode]
        return [target] if target in pop else super().choices(pop, weights, k=k)

def main():
    rows = list(csv.DictReader(open(SRC / "manifest.csv", encoding="utf-8")))
    raw_by_farm = defaultdict(list)
    for r in rows:
        if r["kind"] == "real": raw_by_farm[r["tar"]].append(r["source"])
    all_src = sorted({r["source"] for r in rows if r["kind"] == "real"})
    (OUT / "img").mkdir(parents=True, exist_ok=True); out = []
    for n, r in enumerate(rows):
        dst = OUT / "img" / r["file"]
        if r["kind"] != "composite":
            if not dst.exists():
                try: os.link(SRC / "img" / r["file"], dst)
                except OSError: shutil.copy2(SRC / "img" / r["file"], dst)
            out.append(r); continue
        lr = random.Random(zlib.crc32(f"v4comp|{r['source']}".encode()))
        a = to_np(dist_cover(Image.open(RAW / r["source"]).convert("RGB"), random.Random(zlib.crc32(f"v3|{r['source']}".encode()))))
        # 배경: 다른 tar(다른 개체·품종일 가능성) 우선, 같은 농장코드 폴드 누수 방지를 위해 같은 farm_code 안에서
        same_farm = [s for s in all_src if s != r["source"] and Path(s).name.split("_")[0] == r["farm_code"]]
        oth = lr.choice(same_farm or all_src)
        other = to_np(dist_cover(Image.open(RAW / oth).convert("RGB"), lr))
        if other.shape != a.shape: other = cv2.resize(other, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        fa, mode = compose(a, lr, other)
        dst.write_bytes(jpeg_bytes(to_img(fa)))
        out.append(dict(r, info=mode, other=oth))
        if (n + 1) % 500 == 0: print(n + 1, flush=True)
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    from collections import Counter
    print("rows", len(out), Counter(r["info"] for r in out if r["kind"] == "composite"))

if __name__ == "__main__":
    main()
