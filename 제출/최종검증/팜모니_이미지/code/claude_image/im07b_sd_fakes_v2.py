"""IM07b (v2, 비평 C1·C2·I4 반영): SD1.5 로컬 가중치로 생성 계열 가짜. 라운드로빈 큐(중간에 멈춰도 유형 균형), 재개 가능.
- partial_gen_sd  : dist 처리된 768x1024 사진에서 512 타일을 골라 SD 인페인팅 → 원해상도 그대로 제자리 합성(확대 없음).
                    75%는 마스크 영역만 합성, 25%는 타일 전체(VAE 재구성 흔적) 합성.
- partial_style_sd: 512 타일 img2img(사진풍/화풍 반반) → 마스크 영역만 합성(원해상도).
- full_style_sd   : 전체를 576x768로 줄여 img2img → dist에서 확대(리샘플 대조군 real_resampled가 확대 흔적을 상쇄).
                    사진풍(strength .2~.45)·화풍(.4~.7) 반반.
- ai_gen          : txt2img 576x768 → dist 확대.
붉은 조명 원본 포함. 원본 계보(source/farm_code) 기록. ai_gen은 GEN0~GEN4 다섯 그룹으로 나눠 폴드 쏠림 방지.
사용: python im07b_sd_fakes_v2.py <rounds>
"""
import sys, csv, random, zlib, time
import numpy as np, torch
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from im03_make_fakes import dist_resize, jpeg_bytes, plant_mask, blob_mask, feather, RAW, LOCAL  # noqa
from im07_diffusion_fakes import SCENES, PHOTO, NEG  # noqa

OUT = LOCAL / "im_set_sd2"
ART = ["watercolor painting", "digital illustration", "oil painting", "anime background art", "colored pencil drawing",
       "impressionist painting", "storybook illustration", "3d render", "pastel illustration", "concept art"]
PHOTO_EDIT = ["photo, golden hour lighting", "photo, overcast soft light", "photo, vivid colors, HDR", "photo, cinematic color grading",
              "photo, warm tone", "photo, cool tone", "photo, high contrast", "professional photo, sharp focus"]
INP = ["strawberry leaves", "strawberry flower", "ripe strawberry fruit", "green leaves", "white plastic gutter", "pebbles on the ground",
       "black drip irrigation tube", "yellow sticky trap", "coco peat substrate", "strawberry plant", "metal pipe", "plastic label tag"]
ROUND = [("partial_gen_sd", 8), ("partial_style_sd", 5), ("full_style_sd", 5), ("ai_gen", 6)]
_P = {}

def get(kind):
    from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
    key = "inp" if kind == "partial_gen_sd" else "base"
    if key not in _P:
        for k in list(_P): del _P[k]
        torch.cuda.empty_cache()
        if key == "inp":
            p = StableDiffusionInpaintPipeline.from_pretrained(LOCAL / "weights/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16", safety_checker=None)
            _P["inp"] = p.to("cuda")
        else:
            b = StableDiffusionPipeline.from_pretrained(LOCAL / "weights/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", safety_checker=None).to("cuda")
            _P["base"] = b; _P["i2i"] = StableDiffusionImg2ImgPipeline(**b.components)
        for p in _P.values(): p.enable_attention_slicing(); p.set_progress_bar_config(disable=True)
    return _P

def tile_xy(W, H, lr): return lr.randint(0, W - 512), lr.randint(0, H - 512)

def one(kind, i, man):
    lr = random.Random(zlib.crc32(f"v2|{kind}|{i}".encode()))
    g = torch.Generator("cpu").manual_seed(lr.randint(0, 2**31)); steps = lr.randint(20, 30); cfg = lr.uniform(5, 9)
    P = get(kind)
    if kind == "ai_gen":
        prompt = f"{lr.choice(SCENES)}, {lr.choice(PHOTO)}"; W, H = (576, 768) if lr.random() < .6 else (768, 576)
        out = P["base"](prompt, negative_prompt=NEG, width=W, height=H, num_inference_steps=steps, guidance_scale=cfg, generator=g).images[0]
        return dist_resize(out, lr), dict(source="", group=f"gen{i}", farm_code=f"GEN{i % 5}", tar="GEN"), f"{prompt}|{steps}|{cfg:.1f}"
    r = man[lr.randrange(len(man))]
    base = dist_resize(Image.open(RAW / r["file"]).convert("RGB"), lr); a = np.asarray(base).copy(); W, H = base.size
    meta = dict(source=r["file"], group=r["group"], farm_code=r["farm_code"], tar=r["tar"])
    if kind == "full_style_sd":
        photo = lr.random() < .5
        prompt = f"strawberry plants in a farm, {lr.choice(PHOTO_EDIT if photo else ART)}"; s = lr.uniform(.2, .45) if photo else lr.uniform(.4, .7)
        small = base.resize((W * 3 // 4 // 8 * 8, H * 3 // 4 // 8 * 8), Image.LANCZOS)
        out = P["i2i"](prompt, image=small, strength=s, num_inference_steps=steps, guidance_scale=cfg, negative_prompt=NEG, generator=g).images[0]
        return dist_resize(out, lr), meta, f"{prompt}|s={s:.2f}"
    x0, y0 = tile_xy(W, H, lr); tile = base.crop((x0, y0, x0 + 512, y0 + 512)); ta = a[y0:y0 + 512, x0:x0 + 512]
    if kind == "partial_gen_sd":
        m = blob_mask((512, 512), lr, (.08, .4)); prompt = f"{lr.choice(INP)}, photo, realistic"
        gen = np.asarray(P["inp"](prompt, image=tile, mask_image=Image.fromarray(m * 255), width=512, height=512,
                                  num_inference_steps=steps, guidance_scale=cfg, generator=g).images[0])
        _ = lr.random(); whole = False   # 재비평 중요-4: 타일 전체 합성(사각 경계 지름길) 제거, 난수 소비는 유지
        fm = np.ones((512, 512, 1), np.float32) if whole else feather(m, lr.randint(1, 8))[..., None]
        info = f"{prompt}|area={m.mean():.3f}|tile_whole={int(whole)}"
    else:  # partial_style_sd
        photo = lr.random() < .5; s = lr.uniform(.25, .45) if photo else lr.uniform(.4, .65)
        prompt = f"strawberry plants, {lr.choice(PHOTO_EDIT if photo else ART)}"
        gen = np.asarray(P["i2i"](prompt, image=tile, strength=s, num_inference_steps=steps, guidance_scale=cfg, negative_prompt=NEG, generator=g).images[0])
        m = plant_mask(ta) * blob_mask((512, 512), lr, (.2, .6)) if lr.random() < .6 else blob_mask((512, 512), lr, (.15, .5))
        if m.sum() < .03 * m.size: m = blob_mask((512, 512), lr, (.15, .5))
        fm = feather(m, lr.randint(2, 10))[..., None]; info = f"{prompt}|s={s:.2f}"
    a[y0:y0 + 512, x0:x0 + 512] = np.clip(gen * fm + ta * (1 - fm), 0, 255).astype(np.uint8)
    return Image.fromarray(a), meta, info + f"|tile={x0},{y0}"

def main(rounds):
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    man = list(csv.DictReader(open(RAW / "manifest.csv", encoding="utf-8")))
    rc = LOCAL / "red_sources.txt"   # 사용자 결정(10-10): 빨간 조명 원본 제외
    if rc.exists():
        red = set(rc.read_text(encoding="utf-8").split(chr(10))); man = [r for r in man if r["file"] not in red]
    print("sources", len(man), flush=True)
    mp = OUT / "manifest.csv"; done = set()
    if mp.exists(): done = {r["file"] for r in csv.DictReader(open(mp, encoding="utf-8"))}
    f = open(mp, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["file", "label", "kind", "info", "source", "group", "farm_code", "tar", "other"])
    if not mp.exists() or mp.stat().st_size == 0: pass
    if f.tell() == 0: w.writeheader()
    t = time.time(); cnt = {k: 0 for k, _ in ROUND}
    for rd in range(rounds):
        for kind, k in ROUND:
            for j in range(k):
                i = rd * k + j; fn = f"{kind}_{i:05d}.jpg"
                if fn in done: continue
                try:
                    img, meta, info = one(kind, i, man)
                except Exception as e:
                    print("FAIL", kind, i, repr(e)[:200], flush=True); torch.cuda.empty_cache(); continue
                (OUT / "img" / fn).write_bytes(jpeg_bytes(img))
                w.writerow(dict(file=fn, label=1, kind=kind, info=info, other="", **meta)); f.flush(); cnt[kind] += 1
        print(f"round {rd} {cnt} {time.time()-t:.0f}s", flush=True)

if __name__ == "__main__":
    main(int(sys.argv[1]))
