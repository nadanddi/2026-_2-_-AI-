"""IM03: 실제 원본 → (a) 배포 처리만 한 실제(라벨0) + (b) 프로그램 제작 가짜(라벨1) 1장씩.
배포 처리 dist(): 3:4 크롭(방향 유지, 오프셋 약간 무작위) → 768x1024 리사이즈(보간 무작위) → JPEG Q92 4:2:0 (공개 21장 실측).
가짜 유형(공식 6유형 중 프로그램으로 근사 가능한 것):
  filter_blur / filter_noise / filter_sp / filter_jpeg   (공식 '지정 영상처리 필터')
  composite   : 식물(녹색) 마스크를 다른 실제 사진/단색/흰 여백 배경에 붙임
  full_style  : OpenCV 스타일화·카툰·디테일 강조 (생성 모델 스타일 변환의 약한 근사)
  partial_style: 위 스타일을 일부 영역에만
  partial_inpaint: cv2.inpaint로 일부 영역 재구성 (부분 생성의 약한 근사)
원본 계보(source)를 행마다 기록해 같은 원본이 학습/검증에 갈라지지 않게 함.
"""
import csv, io, random, sys, zlib
import numpy as np, cv2
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent / "local"
RAW = LOCAL / "im_real_raw"
OUT = LOCAL / "im_set_v1"
SEED = 7
RESAMPLE = [Image.LANCZOS, Image.BICUBIC, Image.BILINEAR, Image.BOX]

def crop34(img, rng):
    w, h = img.size
    tw, th = (768, 1024) if h >= w else (1024, 768)
    ar = tw / th
    if w / h > ar:
        nw = int(round(h * ar)); x0 = int((w - nw) * rng.uniform(0.3, 0.7)); img = img.crop((x0, 0, x0 + nw, h))
    else:
        nh = int(round(w / ar)); y0 = int((h - nh) * rng.uniform(0.3, 0.7)); img = img.crop((0, y0, w, y0 + nh))
    return img, (tw, th)

def dist_resize(img, rng):
    img, size = crop34(img, rng)
    return img.resize(size, rng.choice(RESAMPLE))

def jpeg_bytes(img, q=92):
    b = io.BytesIO(); img.save(b, "JPEG", quality=q, subsampling=2); return b.getvalue()

def to_np(img): return np.asarray(img.convert("RGB"))
def to_img(a): return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def plant_mask(a):
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] >= 25) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 50) & (hsv[..., 2] >= 30)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return m

def blob_mask(shape, rng, frac=(0.08, 0.35)):
    h, w = shape[:2]; m = np.zeros((h, w), np.uint8)
    target = rng.uniform(*frac) * h * w
    while m.sum() < target:
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        ax, ay = rng.randint(w // 12, w // 4), rng.randint(h // 12, h // 4)
        cv2.ellipse(m, (cx, cy), (ax, ay), rng.uniform(0, 180), 0, 360, 1, -1)
    return m

def feather(m, r):
    if r <= 0: return m.astype(np.float32)
    k = 2 * r + 1
    return cv2.GaussianBlur(m.astype(np.float32), (k, k), 0)

def stylize(a, rng):
    bgr = cv2.cvtColor(a, cv2.COLOR_RGB2BGR); kind = rng.choice(["styl", "cartoon", "detail", "quant", "painterly"])
    if kind == "styl":
        o = cv2.stylization(bgr, sigma_s=rng.uniform(30, 120), sigma_r=rng.uniform(0.2, 0.6))
    elif kind == "detail":
        o = cv2.detailEnhance(bgr, sigma_s=rng.uniform(10, 40), sigma_r=rng.uniform(0.1, 0.3))
    elif kind == "quant":
        Z = bgr.reshape(-1, 3).astype(np.float32); K = rng.randint(8, 24)
        _, lab, cen = cv2.kmeans(Z, K, None, (cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0), 1, cv2.KMEANS_PP_CENTERS)
        o = cen[lab.flatten()].reshape(bgr.shape).astype(np.uint8); o = cv2.bilateralFilter(o, 7, 50, 50)
    elif kind == "painterly":
        o = bgr
        for _ in range(rng.randint(2, 5)): o = cv2.bilateralFilter(o, 9, rng.uniform(40, 120), rng.uniform(5, 15))
        o = cv2.convertScaleAbs(o, alpha=rng.uniform(1.0, 1.3), beta=rng.uniform(-10, 15))
    else:  # cartoon
        sm = bgr
        for _ in range(3): sm = cv2.bilateralFilter(sm, 9, 75, 75)
        g = cv2.medianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 7)
        e = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        o = cv2.bitwise_and(sm, sm, mask=e)
    # 색감 이동(일러스트풍 채도 강조)
    hsv = cv2.cvtColor(o, cv2.COLOR_BGR2HSV).astype(np.float32); hsv[..., 1] *= rng.uniform(1.0, 1.5)
    o = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(o, cv2.COLOR_BGR2RGB), kind

def make_fake(a, kind, rng, other):
    h, w = a.shape[:2]; info = ""
    if kind == "filter_blur":
        s = rng.uniform(0.8, 3.0); o = cv2.GaussianBlur(a, (0, 0), s); info = f"sigma={s:.2f}"
    elif kind == "filter_noise":
        s = rng.uniform(4, 20); o = a.astype(np.float32) + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, s, a.shape); info = f"sigma={s:.1f}"
    elif kind == "filter_sp":
        p = rng.uniform(0.003, 0.05); g = np.random.default_rng(rng.randint(0, 1 << 30)); r = g.random((h, w))
        o = a.copy(); o[r < p / 2] = 0; o[r > 1 - p / 2] = 255; info = f"p={p:.4f}"
    elif kind == "filter_jpeg":
        q = rng.randint(5, 45); o = to_np(Image.open(io.BytesIO(jpeg_bytes(to_img(a), q)))); info = f"q={q}"
    elif kind == "composite":
        m = plant_mask(a)
        bgk = rng.choice(["other", "other_blur", "solid", "white_margin"])
        if bgk.startswith("other"):
            b = cv2.resize(other, (w, h), interpolation=cv2.INTER_AREA)
            if bgk == "other_blur": b = cv2.GaussianBlur(b, (0, 0), rng.uniform(2, 8))
        else:
            col = np.array([rng.randint(150, 255)] * 3) if bgk == "white_margin" else np.array([rng.randint(0, 255) for _ in range(3)])
            b = np.ones_like(a) * col
        if bgk == "white_margin":
            s = rng.uniform(0.75, 0.92); sw, sh = int(w * s), int(h * s)
            small = cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA); ms = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST)
            x0, y0 = rng.randint(0, w - sw), rng.randint(0, h - sh)
            canvas = np.full_like(a, 255); canvas[y0:y0 + sh, x0:x0 + sw] = small
            mm = np.zeros((h, w), np.uint8); mm[y0:y0 + sh, x0:x0 + sw] = 1
            pl = np.zeros((h, w), np.uint8); pl[y0:y0 + sh, x0:x0 + sw] = ms
            b2 = np.where(mm[..., None] > 0, b, 255)
            f = feather(pl, rng.randint(0, 3))[..., None]; o = canvas * f + b2 * (1 - f)
        else:
            f = feather(m, rng.randint(0, 3))[..., None]
            o = a * f + b * (1 - f)
            if rng.random() < 0.5:  # 작물 밝기·대비 차
                o = o * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
        info = bgk
    elif kind == "full_style":
        o, info = stylize(a, rng)
    elif kind == "partial_style":
        st, info = stylize(a, rng)
        m = plant_mask(a) * blob_mask(a.shape, rng, (0.15, 0.5)) if rng.random() < 0.6 else blob_mask(a.shape, rng)
        if m.sum() < 0.03 * h * w: m = blob_mask(a.shape, rng)
        f = feather(m, rng.randint(2, 12))[..., None]; o = st * f + a * (1 - f)
    elif kind == "partial_inpaint":
        m = blob_mask(a.shape, rng, (0.03, 0.15))
        o = cv2.inpaint(a, m, rng.choice([3, 7, 15]), rng.choice([cv2.INPAINT_TELEA, cv2.INPAINT_NS]))
        # 인페인트 영역에 약한 질감 회복(평평함 완화)
        g = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(1, 5), a.shape)
        o = o.astype(np.float32) + g * m[..., None]; info = f"area={m.mean():.3f}"
    return np.clip(o, 0, 255).astype(np.uint8), info

KINDS = ["filter_blur", "filter_noise", "filter_sp", "filter_jpeg", "composite", "composite",
         "full_style", "full_style", "partial_style", "partial_style", "partial_inpaint", "partial_inpaint"]

def main():
    rng = random.Random(SEED)
    man = list(csv.DictReader(open(RAW / "manifest.csv", encoding="utf-8")))
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "img").mkdir(exist_ok=True)
    rows = []; done = set()
    mp = OUT / "manifest.csv"
    if mp.exists():
        rows = list(csv.DictReader(open(mp, encoding="utf-8"))); done = {r["source"] for r in rows}
    order = list(range(len(man))); rng.shuffle(order)
    new = 0
    for n, i in enumerate(order):
        r = man[i]; src = r["file"]
        if src in done: continue
        lr = random.Random(zlib.crc32(f'{SEED}|{src}'.encode()))
        img = Image.open(RAW / src).convert("RGB")
        base = dist_resize(img, lr)
        a = to_np(base)
        sid = f"s{zlib.crc32(src.encode()):010d}"
        # 실제
        (OUT / "img" / f"{sid}_real.jpg").write_bytes(jpeg_bytes(base))
        # 가짜
        kind = KINDS[(n + lr.randint(0, 100)) % len(KINDS)]
        oth = man[lr.randrange(len(man))]["file"]
        other = to_np(dist_resize(Image.open(RAW / oth).convert("RGB"), lr))
        fake, info = make_fake(a, kind, lr, other)
        (OUT / "img" / f"{sid}_fake.jpg").write_bytes(jpeg_bytes(to_img(fake)))
        for lab, fn, k, inf in [(0, f"{sid}_real.jpg", "real", ""), (1, f"{sid}_fake.jpg", kind, info)]:
            rows.append(dict(file=fn, label=lab, kind=k, info=inf, source=src, group=r["group"], farm_code=r["farm_code"],
                             tar=r["tar"], other=oth if k == "composite" else ""))
        new += 1
        if new % 100 == 0: print(new, flush=True)
    with open(mp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("total rows", len(rows), "new sources", new)

if __name__ == "__main__":
    main()
