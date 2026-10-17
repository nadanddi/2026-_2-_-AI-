"""IM15: 가짜 제작 5판 — 공개 가짜와 그 AI허브 원본을 직접 비교해 측정한 운영진 변형 방식(IM10·IM12) 그대로. 서버(RTX 6000 Ada) 실행용.
측정 근거(공개 샘플 ↔ 원본):
 - 합성 bpem: 사진 그대로 + 다른 사진의 잎 몇 장을 오려 붙임 / yuag: 줄기째 식물 여러 포기를 붙임 + 흰 여백
 - 블러 vfjx: 배포 크기에서 가우시안 σ≈1.5 (원본 해상도 적용보다 잘 맞음)
 - 소금후추 jrtg: 원본 해상도에서 찍고 축소(뚜렷한 점 없음) → 3판 방식 유지
 - 전체 스타일 ttmn: 구도·형태 유지, 사진 전체 질감·명암 재표현(원본 대비 평균 차 16)
 - 부분 스타일 xeeo: 식물 한 포기(물체 모양)만 변형
 - 부분 생성 uexm: 사진 전체가 약하게 다시 그려지고(중앙 차 11.7) 일부 영역이 크게 바뀜
유형과 방법:
 filter_blur(배포 크기 σ 1.0~2.2) / filter_noise·filter_sp·filter_jpeg(3판 원본 해상도 방식)
 composite_v5(잎·포기 오려 붙이기 1~6개, 화면 10~40%, 30%는 흰 여백)
 full_style_xl(SDXL img2img 원해상도, strength .30~.55, 사진풍/화풍 반반)
 partial_style_xl(같은 img2img 결과를 식물 한 포기 마스크에만)
 partial_gen_xl(전체 img2img strength .12~.25로 약하게 재표현 → 그 위에 SDXL 인페인팅으로 물체/덩어리 영역 재생성)
 ai_gen_xl(SDXL txt2img 832x1216/1216x832 → 배포 처리 축소)
원본: im_real_raw 중 빨간 조명 제외 전부(농장 고르게). 배포 처리 = 덮게 축소 → 가운데 크롭 → JPEG Q92.
사용: python im15_fakes_v5.py <kind> <n> [gpu=0]   (출력 local/im_set_v5/<kind>_*.jpg, manifest 이어쓰기, 재개 가능)
"""
import sys, csv, io, random, zlib, time, os
import numpy as np, cv2
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im03_make_fakes import jpeg_bytes, to_np, to_img, feather, plant_mask, blob_mask, RAW, LOCAL  # noqa
from im03c_make_fakes_v3 import dist_cover, raw_filter  # noqa

OUT = LOCAL / "im_set_v5"
SCENES = ["strawberry plants growing in white hydroponic gutters in a vertical farm", "strawberry seedlings in black pots on a greenhouse bench",
          "close-up of strawberry leaves and white flowers in a greenhouse", "rows of strawberry plants on elevated hydroponic troughs, drip irrigation tubes",
          "strawberry plants with red ripe fruits hanging from a raised bed", "strawberry plant in a grow bag with coco peat substrate, yellow sticky trap",
          "strawberry runners and leaves on a white plastic tray, pebbles on the floor", "indoor strawberry farm under white LED light, metal frames, white gutters"]
PHOTO = ["photo, realistic, smartphone photo", "DSLR photo, natural light, high detail", "realistic photograph", "phone camera snapshot"]
ART = ["watercolor painting", "digital illustration", "oil painting", "anime background art", "colored pencil drawing", "storybook illustration", "painterly"]
PEDIT = ["photo, cinematic color grading", "photo, warm tone", "photo, vivid colors, HDR", "photo, soft light", "professional photo, sharp focus"]
INP = ["strawberry leaves", "strawberry flower", "ripe strawberry fruit", "green leaves", "white plastic gutter", "pebbles", "black drip irrigation tube",
       "yellow sticky trap", "coco peat substrate", "strawberry plant", "plastic label tag"]
NEG = "text, watermark, blurry, lowres, deformed"
_P = {}

def rawp(f):   # 윈도우에서 만든 manifest의 '' 경로를 리눅스(서버)에서도 열리게
    return RAW / f.replace("\\", "/")

def pipes(which):
    import torch
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, AutoencoderKL
    if which in _P: return _P[which]
    W = LOCAL / "weights"
    if which == "xl":
        vae = AutoencoderKL.from_pretrained(W / "sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        b = StableDiffusionXLPipeline.from_pretrained(W / "stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16, variant="fp16").to("cuda")
        _P["xl"] = (b, StableDiffusionXLImg2ImgPipeline(**b.components))
    else:
        _P["inp"] = StableDiffusionXLInpaintPipeline.from_pretrained(W / "sdxl-inpainting", torch_dtype=torch.float16, variant="fp16").to("cuda")
    for p in (_P[which] if isinstance(_P[which], tuple) else (_P[which],)): p.set_progress_bar_config(disable=True)
    return _P[which]

def sources():
    red = set((LOCAL / "red_sources.txt").read_text(encoding="utf-8").split(chr(10))) - {""}
    return [r for r in csv.DictReader(open(RAW / "manifest.csv", encoding="utf-8")) if r["file"] not in red]

def object_mask(a, lr, lo=.08, hi=.35):
    m = plant_mask(a); n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    comps = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > .003 * m.size]; lr.shuffle(comps); out = np.zeros_like(m); target = lr.uniform(lo, hi)
    for i in comps:
        out[lab == i] = 1
        if out.mean() >= target: break
    if out.mean() > target * 1.5:
        for _ in range(20):
            mm = out * blob_mask(out.shape, lr, (target * .8, target * 1.2))
            if .5 * target <= mm.mean() <= 1.5 * target: out = mm; break
    return out if out.mean() > .02 else blob_mask(a.shape, lr, (lo, hi))

def composite_v5(a, other, lr, maxfrac=1.0, erode=0):
    h, w = a.shape[:2]; mo = plant_mask(other)
    if erode:   # v5b: 잎 사이를 끊어 덩어리를 잎 단위로 쪼갬(사각 덩어리 방지)
        mo = cv2.erode(mo, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mo, 8)
    comps = [i for i in range(1, n) if .002 * mo.size < st[i, cv2.CC_STAT_AREA] < maxfrac * mo.size]; lr.shuffle(comps)
    o = a.astype(np.float32).copy(); target = lr.uniform(.10, .40); cover = np.zeros((h, w), np.uint8); k = 0
    for ci in comps[:12]:
        x, y, cw, ch = st[ci, :4]; m = (lab[y:y + ch, x:x + cw] == ci).astype(np.uint8)
        s = lr.uniform(.7, 1.2); cw2, ch2 = max(16, int(cw * s)), max(16, int(ch * s))
        if cw2 >= w or ch2 >= h: continue
        patch = cv2.resize(other[y:y + ch, x:x + cw], (cw2, ch2), interpolation=cv2.INTER_AREA); m2 = cv2.resize(m, (cw2, ch2), interpolation=cv2.INTER_NEAREST)
        x0, y0 = lr.randint(0, w - cw2), lr.randint(0, h - ch2)
        if erode: m2 = cv2.dilate(m2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, erode // 2) | 1,) * 2))
        f = feather(m2, lr.randint(0, 2))[..., None]; o[y0:y0 + ch2, x0:x0 + cw2] = patch * f + o[y0:y0 + ch2, x0:x0 + cw2] * (1 - f)
        cover[y0:y0 + ch2, x0:x0 + cw2] |= m2; k += 1
        if cover.mean() >= target or k >= 6: break
    o = np.clip(o, 0, 255).astype(np.uint8); info = f"paste{k}|cover={cover.mean():.2f}"
    if lr.random() < .3:   # yuag형 흰 여백
        s = lr.uniform(.85, .96); sw, sh = int(w * s), int(h * s); small = cv2.resize(o, (sw, sh), interpolation=cv2.INTER_AREA)
        c = np.full_like(o, 255); x0, y0 = lr.randint(0, w - sw), lr.randint(0, h - sh); c[y0:y0 + sh, x0:x0 + sw] = small; o = c; info += "|margin"
    return o, info

def leaf_mask(other, lr, lo=.10, hi=.35):
    """잎 단위 마스크: 약한 침식으로 잎 사이를 끊고, 면적 .3~8% 덩어리만 모아 목표 면적까지 → 다시 팽창. 타원 대체 없음."""
    mo = plant_mask(other); k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    me = cv2.erode(mo, k5, iterations=2); n, lab, st, _ = cv2.connectedComponentsWithStats(me, 8)
    comps = [i for i in range(1, n) if .003 * me.size < st[i, cv2.CC_STAT_AREA] < .08 * me.size]; lr.shuffle(comps)
    out = np.zeros_like(mo); target = lr.uniform(lo, hi)
    for i in comps:
        out[lab == i] = 1
        if out.mean() >= target: break
    return cv2.dilate(out, k5, iterations=2) & mo

def composite_v5b(a, others, lr):
    """v5b: 다른 사진(최대 6장 시도)에서 잎 단위 영역(10~35%)을 골라 붙임. 50%는 위치 이동, 30%는 흰 여백."""
    h, w = a.shape[:2]
    for other in others:
        m = leaf_mask(other, lr)
        if m.mean() >= .06: break
    else:
        return None, "no_leaves"
    src = other
    if lr.random() < .5:
        dx, dy = lr.randint(-w // 6, w // 6), lr.randint(-h // 6, h // 6); M = np.float32([[1, 0, dx], [0, 1, dy]])
        src = cv2.warpAffine(other, M, (w, h), borderMode=cv2.BORDER_REFLECT); m = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST)
    f = feather(m, lr.randint(1, 2))[..., None]; o = np.clip(src * f + a * (1 - f), 0, 255).astype(np.uint8); info = f"leafpaste|cover={m.mean():.2f}"
    if lr.random() < .3:
        s = lr.uniform(.85, .96); sw, sh = int(w * s), int(h * s); small = cv2.resize(o, (sw, sh), interpolation=cv2.INTER_AREA)
        c = np.full_like(o, 255); x0, y0 = lr.randint(0, w - sw), lr.randint(0, h - sh); c[y0:y0 + sh, x0:x0 + sw] = small; o = c; info += "|margin"
    return o, info

def make(kind, i, man):
    import torch
    lr = random.Random(zlib.crc32(f"v5|{kind}|{i}".encode())); g = torch.Generator("cpu").manual_seed(lr.randint(0, 2**31))
    r = man[lr.randrange(len(man))]; raw = Image.open(rawp(r["file"])).convert("RGB"); st = lr.getstate(); base = dist_cover(raw, lr); a = to_np(base)
    meta = dict(source=r["file"], group=r["group"], farm_code=r["farm_code"], tar=r["tar"])
    if kind == "filter_blur":
        s = lr.uniform(1.0, 2.2); return to_img(cv2.GaussianBlur(a, (0, 0), s)), meta, f"dist_sigma={s:.2f}"
    if kind in ("filter_noise", "filter_sp", "filter_jpeg"):
        fr, info = raw_filter(raw, kind, lr); ls = random.Random(); ls.setstate(st); return dist_cover(fr, ls), meta, info
    if kind in ("composite_v5", "composite_v5b"):
        pool = [x for x in man if x["farm_code"] == r["farm_code"] and x["file"] != r["file"]] or man
        oth = lr.choice(pool); other = to_np(dist_cover(Image.open(rawp(oth["file"])).convert("RGB"), lr))
        if other.shape != a.shape: other = cv2.resize(other, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        if kind == "composite_v5": o, info = composite_v5(a, other, lr)
        else:
            others = [other]
            for _ in range(5):
                ox = lr.choice(pool); t2 = to_np(dist_cover(Image.open(rawp(ox["file"])).convert("RGB"), lr))
                others.append(t2 if t2.shape == a.shape else cv2.resize(t2, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA))
            o, info = composite_v5b(a, others, lr)
            if o is None: raise RuntimeError("잎 영역 부족")
        return to_img(o), dict(meta, other=oth["file"]), info
    steps = lr.randint(25, 35); cfg = lr.uniform(5, 8)
    if kind == "ai_gen_xl":
        base_p, _ = pipes("xl"); prompt = f"{lr.choice(SCENES)}, {lr.choice(PHOTO)}"; W, H = (832, 1216) if lr.random() < .6 else (1216, 832)
        out = base_p(prompt, negative_prompt=NEG, width=W, height=H, num_inference_steps=steps, guidance_scale=cfg, generator=g).images[0]
        return dist_cover(out, lr), dict(source="", group=f"gen{i}", farm_code=f"GEN{i % 5}", tar="GEN"), prompt
    _, i2i = pipes("xl")
    if kind in ("full_style_xl", "partial_style_xl"):
        photo = lr.random() < .5; s = lr.uniform(.30, .45) if photo else lr.uniform(.40, .55)
        prompt = f"strawberry plants in a farm, {lr.choice(PEDIT if photo else ART)}"
        out = np.asarray(i2i(prompt, image=base, strength=s, num_inference_steps=steps, guidance_scale=cfg, negative_prompt=NEG, generator=g).images[0].resize(base.size, Image.LANCZOS))
        if kind == "full_style_xl": return to_img(out), meta, f"{prompt}|s={s:.2f}"
        m = object_mask(a, lr); f = feather(m, lr.randint(1, 6))[..., None]
        return to_img(out * f + a * (1 - f)), meta, f"{prompt}|s={s:.2f}|area={m.mean():.2f}"
    if kind == "partial_gen_xl":
        s0 = lr.uniform(.12, .25)
        glob = i2i("strawberry plants, photo, realistic", image=base, strength=s0, num_inference_steps=steps, guidance_scale=cfg, negative_prompt=NEG, generator=g).images[0].resize(base.size, Image.LANCZOS)
        inp = pipes("inp"); m = object_mask(a, lr, .05, .30) if lr.random() < .5 else blob_mask(a.shape, lr, (.05, .30))
        prompt = f"{lr.choice(INP)}, photo, realistic"
        out = inp(prompt, image=glob, mask_image=Image.fromarray(m * 255), width=base.size[0], height=base.size[1], strength=lr.uniform(.85, 1.0),
                  num_inference_steps=steps, guidance_scale=cfg, generator=g).images[0].resize(base.size, Image.LANCZOS)
        return out, meta, f"{prompt}|global_s={s0:.2f}|area={m.mean():.2f}"
    raise ValueError(kind)

def main(kind, n, gpu="0"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    (OUT / "img").mkdir(parents=True, exist_ok=True); man = sources()
    mp = OUT / f"manifest_{kind}.csv"; done = {r["file"] for r in csv.DictReader(open(mp, encoding="utf-8"))} if mp.exists() else set()
    f = open(mp, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["file", "label", "kind", "info", "source", "group", "farm_code", "tar", "other"])
    if f.tell() == 0: w.writeheader()
    t = time.time(); k = 0
    for i in range(n):
        fn = f"{kind}_{i:05d}.jpg"
        if fn in done: continue
        try: img, meta, info = make(kind, i, man)
        except Exception as e: print("FAIL", i, repr(e)[:200], flush=True); continue
        (OUT / "img" / fn).write_bytes(jpeg_bytes(img)); w.writerow(dict(file=fn, label=1, kind=kind, info=info, other=meta.pop("other", ""), **meta)); f.flush(); k += 1
        if k % 50 == 0: print(f"{kind} {k} {time.time()-t:.0f}s", flush=True)
    print("done", kind, k, f"{time.time()-t:.0f}s")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "0")
