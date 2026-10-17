"""IM03c (v3): v2(im03b)에서 바꾼 점만
- 필터 4종을 **원본 해상도(AI허브 원본, 약 1500x2000)에서 적용한 뒤** dist(축소·크롭·Q92).
  근거: 팀원 v5 측정(운영진 소금후추 0.28%, 노이즈 인접상관 +.23)과 우리 측정(공개 jrtg의 잔차 첨도 2.2·극단비 .003 = 축소로 뭉개진 점잡음).
  강도: blur σ_raw 1~5, 잡음 σ_raw 6~30, 소금후추 p_raw .005~.06, JPEG q_raw 10~60.
- 빨간 조명 원본 제외(사용자 결정 10-10).
- dist는 '짧은 변 768/긴 변 1024를 덮게 축소 → 가운데 크롭'(팀원 v5가 qube 원본과 대조해 확인한 방식)으로 고정, 보간은 LANCZOS/BICUBIC만.
- 리샘플 대조군(실제, 라벨0)은 SD 확대 가짜 수와 맞추기 위해 v3에서는 15%로 낮춤(SD 누적 후 학습 가중으로 최종 맞춤).
- 색·톤만 바꾼 photo_style 가짜는 kind 이름을 'pstyle_color'로 따로 표시(넣고/빼고 비교용, 재비평 중요-3).
그 외(합성·부분 스타일)는 v2와 같음.
"""
import csv, io, random, zlib, sys
import numpy as np, cv2
from pathlib import Path
from PIL import Image
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im03_make_fakes import jpeg_bytes, to_np, to_img, RAW, LOCAL  # noqa
import im03b_make_fakes_v2 as v2  # noqa

OUT = LOCAL / "im_set_v3"
RESAMPLE = [Image.LANCZOS, Image.BICUBIC]

def dist_cover(img, rng):
    w, h = img.size; tw, th = (768, 1024) if h >= w else (1024, 768)
    s = max(tw / w, th / h); nw, nh = max(tw, round(w * s)), max(th, round(h * s))
    r = img.resize((nw, nh), rng.choice(RESAMPLE)); x0, y0 = (nw - tw) // 2, (nh - th) // 2
    return r.crop((x0, y0, x0 + tw, y0 + th))

def raw_filter(raw, kind, rng):
    a = to_np(raw); h, w = a.shape[:2]; g = np.random.default_rng(rng.randint(0, 1 << 30))
    if kind == "filter_blur":
        s = rng.uniform(1, 5); return to_img(cv2.GaussianBlur(a, (0, 0), s)), f"raw_sigma={s:.2f}"
    if kind == "filter_noise":
        s = rng.uniform(6, 30); n = g.normal(0, s, a.shape) if rng.random() < .7 else np.repeat(g.normal(0, s, (h, w))[..., None], 3, 2)
        return to_img(a + n), f"raw_sigma={s:.1f}"
    if kind == "filter_sp":
        p = rng.uniform(.005, .06); r = g.random((h, w)); o = a.copy(); o[r < p / 2] = 0; o[r > 1 - p / 2] = 255
        return to_img(o), f"raw_p={p:.4f}"
    if kind == "filter_jpeg":
        q = rng.randint(10, 60); return Image.open(io.BytesIO(jpeg_bytes(raw, q))).convert("RGB"), f"raw_q={q}"
    raise ValueError(kind)

KINDS = ["filter_blur", "filter_noise", "filter_sp", "filter_jpeg", "composite", "composite", "composite",
         "full_style", "full_style", "partial_style", "partial_style", "partial_style"]

def main():
    red = set((LOCAL / "red_sources.txt").read_text(encoding="utf-8").split(chr(10))) - {""}
    man = [r for r in csv.DictReader(open(RAW / "manifest.csv", encoding="utf-8")) if r["file"] not in red]
    by_farm = defaultdict(list)
    for r in man: by_farm[r["farm_code"]].append(r["file"])
    (OUT / "img").mkdir(parents=True, exist_ok=True); rows = []
    for n, r in enumerate(man):
        src = r["file"]; lr = random.Random(zlib.crc32(f"v3|{src}".encode())); sid = f"s{zlib.crc32(src.encode()):010d}"
        raw = Image.open(RAW / src).convert("RGB")
        st = lr.getstate(); base = dist_cover(raw, lr); a = to_np(base)
        (OUT / "img" / f"{sid}_real.jpg").write_bytes(jpeg_bytes(base)); out = [(0, f"{sid}_real.jpg", "real", "")]
        if lr.random() < .15:
            w0, h0 = base.size; sm = base.resize((w0 * 3 // 4, h0 * 3 // 4), Image.LANCZOS)
            (OUT / "img" / f"{sid}_realrs.jpg").write_bytes(jpeg_bytes(dist_cover(sm, lr))); out.append((0, f"{sid}_realrs.jpg", "real_resampled", "x0.75->up"))
        kind = KINDS[lr.randrange(len(KINDS))]
        if kind.startswith("filter"):
            fr, info = raw_filter(raw, kind, lr)
            ls = random.Random(); ls.setstate(st); fake = dist_cover(fr, ls)   # 실제와 같은 보간·크롭
            fa = to_np(fake)
        else:
            pool = [f for f in by_farm[r["farm_code"]] if f != src] or [src]
            oth = lr.choice(pool); other = to_np(dist_cover(Image.open(RAW / oth).convert("RGB"), lr))
            if other.shape != a.shape: other = cv2.resize(other, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
            fa, info = v2.make_fake(a, kind, lr, other)
            if kind == "full_style": kind = "pstyle_color"
        (OUT / "img" / f"{sid}_fake.jpg").write_bytes(jpeg_bytes(to_img(fa))); out.append((1, f"{sid}_fake.jpg", kind, info))
        for lab, fn, k, inf in out:
            rows.append(dict(file=fn, label=lab, kind=k, info=inf, source=src, group=r["group"], farm_code=r["farm_code"], tar=r["tar"], other=""))
        if (n + 1) % 200 == 0: print(n + 1, flush=True)
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print("rows", len(rows))

if __name__ == "__main__":
    main()
