"""IM03b (v2, 비평 C1·C3·I1·I2·경미 반영): 프로그램 제작 가짜 — 사진처럼 보이는 미묘한 변형 위주.
변경점 (v1 대비):
- 붉은 조명 원본도 포함(제외 안 함).
- 필터 약하게+넓게: blur σ .5~2.5, 잡음 σ 2~14, 소금후추 p .001~.03, JPEG q 15~80.
- composite: 다른 실제 사진의 식물 덩어리를 붙여넣기(크기·위치 무작위, 색 맞춤, 페더) / 원 식물을 흐린·탈색 배경 위로(bpem형) / 드물게 흰 여백.
- full_style(사진풍): LAB 색 전이, 톤 곡선, 채도·색상 이동, 약한 디테일 강조·평활 조합. 만화·색 양자화 제거.
- partial_style: 위 사진풍 변형을 일부 영역에만.
- cv2 inpaint(+국소 잡음) 제거 → 부분 생성은 SD(im07b)에서.
- 리샘플 대조군: 실제 사진 일부를 576x768로 줄였다 dist에서 확대(라벨 0) → SD 가짜의 '확대 흔적' 지름길 무력화.
- composite 배경(other)은 같은 farm_code 안에서만 골라 폴드 누수 제거.
"""
import csv, io, random, zlib
import numpy as np, cv2
from pathlib import Path
from PIL import Image
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im03_make_fakes import dist_resize, jpeg_bytes, to_np, to_img, plant_mask, blob_mask, feather, RAW, LOCAL  # noqa

OUT = LOCAL / "im_set_v2"

def lab_transfer(a, ref, amt):
    A = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32); R = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    ma, sa = A.reshape(-1, 3).mean(0), A.reshape(-1, 3).std(0) + 1e-3; mr, sr = R.reshape(-1, 3).mean(0), R.reshape(-1, 3).std(0) + 1e-3
    T = (A - ma) / sa * sr + mr
    O = A * (1 - amt) + T * amt
    return cv2.cvtColor(np.clip(O, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

def photo_style(a, rng, ref):
    ops = rng.sample(["lab", "tone", "hsv", "detail", "smooth", "clahe"], rng.randint(2, 4)); o = a.copy()
    for op in ops:
        if op == "lab": o = lab_transfer(o, ref, rng.uniform(.4, 1.0))
        elif op == "tone":
            g = rng.uniform(.7, 1.4); c = rng.uniform(.85, 1.25)
            x = (o.astype(np.float32) / 255) ** g; x = (x - .5) * c + .5 + rng.uniform(-.06, .06); o = np.clip(x * 255, 0, 255).astype(np.uint8)
        elif op == "hsv":
            h = cv2.cvtColor(o, cv2.COLOR_RGB2HSV).astype(np.float32)
            h[..., 0] = (h[..., 0] + rng.uniform(-8, 8)) % 180; h[..., 1] *= rng.uniform(.7, 1.4); h[..., 2] *= rng.uniform(.9, 1.15)
            o = cv2.cvtColor(np.clip(h, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        elif op == "detail":
            o = cv2.cvtColor(cv2.detailEnhance(cv2.cvtColor(o, cv2.COLOR_RGB2BGR), sigma_s=rng.uniform(5, 20), sigma_r=rng.uniform(.05, .2)), cv2.COLOR_BGR2RGB)
        elif op == "smooth":
            o = cv2.bilateralFilter(o, rng.choice([5, 7, 9]), rng.uniform(20, 60), rng.uniform(3, 9))
        elif op == "clahe":
            L = cv2.cvtColor(o, cv2.COLOR_RGB2LAB); cl = cv2.createCLAHE(clipLimit=rng.uniform(1.5, 3.5), tileGridSize=(8, 8))
            L[..., 0] = cl.apply(L[..., 0]); o = cv2.cvtColor(L, cv2.COLOR_LAB2RGB)
    return o, "+".join(ops)

def components(m, min_frac=.01):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > min_frac * m.size], lab, st

def make_fake(a, kind, rng, other):
    h, w = a.shape[:2]
    if kind == "filter_blur":
        s = rng.uniform(.5, 2.5); return cv2.GaussianBlur(a, (0, 0), s), f"sigma={s:.2f}"
    if kind == "filter_noise":
        s = rng.uniform(2, 14); g = np.random.default_rng(rng.randint(0, 1 << 30))
        n = g.normal(0, s, a.shape) if rng.random() < .7 else np.repeat(g.normal(0, s, a.shape[:2])[..., None], 3, 2)
        return np.clip(a + n, 0, 255).astype(np.uint8), f"sigma={s:.1f}"
    if kind == "filter_sp":
        p = rng.uniform(.001, .03); g = np.random.default_rng(rng.randint(0, 1 << 30)); r = g.random((h, w))
        o = a.copy(); o[r < p / 2] = 0; o[r > 1 - p / 2] = 255; return o, f"p={p:.4f}"
    if kind == "filter_jpeg":
        q = rng.randint(15, 80); return to_np(Image.open(io.BytesIO(jpeg_bytes(to_img(a), q)))), f"q={q}"
    if kind == "composite":
        mode = rng.choices(["paste", "bg_swap", "margin"], [.55, .35, .10])[0]
        if mode == "paste":   # other의 식물 덩어리를 a 위에
            mo = plant_mask(other); comps, lab, st = components(mo)
            if not comps: mode = "bg_swap"
            else:
                o = a.astype(np.float32).copy(); k = rng.randint(1, min(3, len(comps)))
                for ci in rng.sample(comps, k):
                    x, y, cw, ch = st[ci, :4]; m = (lab[y:y + ch, x:x + cw] == ci).astype(np.uint8)
                    s = rng.uniform(.6, 1.2); cw2, ch2 = max(8, int(cw * s)), max(8, int(ch * s))
                    if cw2 >= w or ch2 >= h: continue
                    patch = cv2.resize(other[y:y + ch, x:x + cw], (cw2, ch2), interpolation=cv2.INTER_AREA); m2 = cv2.resize(m, (cw2, ch2), interpolation=cv2.INTER_NEAREST)
                    x0, y0 = rng.randint(0, w - cw2), rng.randint(0, h - ch2)
                    if rng.random() < .7: patch = lab_transfer(patch, a[y0:y0 + ch2, x0:x0 + cw2], rng.uniform(.3, .8))
                    f = feather(m2, rng.randint(0, 3))[..., None]
                    o[y0:y0 + ch2, x0:x0 + cw2] = patch * f + o[y0:y0 + ch2, x0:x0 + cw2] * (1 - f)
                return np.clip(o, 0, 255).astype(np.uint8), f"paste{k}"
        m = plant_mask(a)
        if mode == "bg_swap":
            b = other if rng.random() < .6 else a
            b = cv2.GaussianBlur(b, (0, 0), rng.uniform(1.5, 6))
            if rng.random() < .5:
                gb = cv2.cvtColor(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB); b = (b * .4 + gb * .6).astype(np.uint8)
            f = feather(m, rng.randint(0, 3))[..., None]
            plant = a if rng.random() < .6 else photo_style(a, rng, other)[0]
            return np.clip(plant * f + b * (1 - f), 0, 255).astype(np.uint8), "bg_swap"
        s = rng.uniform(.8, .95); sw, sh = int(w * s), int(h * s)
        small = cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA); x0, y0 = rng.randint(0, w - sw), rng.randint(0, h - sh)
        o = np.full_like(a, 255); o[y0:y0 + sh, x0:x0 + sw] = small; return o, "margin"
    if kind == "full_style":
        return photo_style(a, rng, other)
    if kind == "partial_style":
        st, info = photo_style(a, rng, other)
        m = plant_mask(a) * blob_mask(a.shape, rng, (.1, .45)) if rng.random() < .6 else blob_mask(a.shape, rng, (.05, .35))
        if m.sum() < .02 * m.size: m = blob_mask(a.shape, rng, (.05, .35))
        f = feather(m, rng.randint(1, 10))[..., None]
        return np.clip(st * f + a * (1 - f), 0, 255).astype(np.uint8), info
    raise ValueError(kind)

KINDS = ["filter_blur", "filter_noise", "filter_sp", "filter_jpeg", "composite", "composite", "composite",
         "full_style", "full_style", "partial_style", "partial_style", "partial_style"]

def main():
    man = list(csv.DictReader(open(RAW / "manifest.csv", encoding="utf-8")))
    by_farm = defaultdict(list)
    for r in man: by_farm[r["farm_code"]].append(r["file"])
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    rows = []
    for n, r in enumerate(man):
        src = r["file"]; lr = random.Random(zlib.crc32(f"v2|{src}".encode()))
        sid = f"s{zlib.crc32(src.encode()):010d}"
        img = Image.open(RAW / src).convert("RGB"); base = dist_resize(img, lr); a = to_np(base)
        (OUT / "img" / f"{sid}_real.jpg").write_bytes(jpeg_bytes(base))
        out = [(0, f"{sid}_real.jpg", "real", "")]
        if lr.random() < .25:  # 리샘플 대조군(실제)
            w0, h0 = base.size; sm = base.resize((w0 * 3 // 4, h0 * 3 // 4), Image.LANCZOS)
            (OUT / "img" / f"{sid}_realrs.jpg").write_bytes(jpeg_bytes(dist_resize(sm, lr)))
            out.append((0, f"{sid}_realrs.jpg", "real_resampled", "x0.75->up"))
        kind = KINDS[lr.randrange(len(KINDS))]
        pool = [f for f in by_farm[r["farm_code"]] if f != src] or [src]
        oth = lr.choice(pool); other = to_np(dist_resize(Image.open(RAW / oth).convert("RGB"), lr))
        if other.shape != a.shape: other = cv2.resize(other, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        fake, info = make_fake(a, kind, lr, other)
        (OUT / "img" / f"{sid}_fake.jpg").write_bytes(jpeg_bytes(to_img(fake)))
        out.append((1, f"{sid}_fake.jpg", kind, info))
        for lab, fn, k, inf in out:
            rows.append(dict(file=fn, label=lab, kind=k, info=inf, source=src, group=r["group"], farm_code=r["farm_code"], tar=r["tar"],
                             other=oth if k == "composite" else ""))
        if (n + 1) % 200 == 0: print(n + 1, flush=True)
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print("rows", len(rows))

if __name__ == "__main__":
    main()
