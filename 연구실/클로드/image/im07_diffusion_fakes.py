"""IM07: 확산 모델(SD1.5, 로컬 가중치)로 생성 계열 가짜 제작 — 공식 유형 중 전체 생성·전체 스타일·부분 생성·부분 스타일의 근사.
  ai_gen        : txt2img (딸기 시설재배 장면 프롬프트)
  full_style_sd : img2img (실제 사진 → 일러스트/수채화/유화 등 스타일, strength .45~.75)
  partial_gen_sd: 인페인팅 모델로 일부 영역 재생성(사실적 프롬프트) 후 원본에 마스크 합성
  partial_style_sd: img2img 스타일 결과를 일부 영역에만 합성
원본(실제)은 im_set_v1과 같은 원본에서 가져오되 결과 manifest에 source를 기록 → 같은 계보는 같은 폴드.
GPU 1050Ti(4GB): fp16 가중치 + 순차 오프로딩, 512 기준 생성 후 dist 처리(3:4 크롭·리사이즈·JPEG Q92).
사용: python im07_diffusion_fakes.py <kind> <n> [start]
"""
import sys, csv, random, zlib, time, io
import numpy as np, torch
from pathlib import Path
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent / "local"
sys.path.insert(0, str(HERE))
from im03_make_fakes import dist_resize, jpeg_bytes, blob_mask, plant_mask, feather, RAW  # noqa

OUT = LOCAL / "im_set_sd"
SCENES = ["strawberry plants growing in white hydroponic gutters in a vertical farm", "strawberry seedlings in black pots on a greenhouse bench",
          "close-up of strawberry leaves and white flowers in a greenhouse", "rows of strawberry plants on elevated hydroponic troughs, drip irrigation tubes",
          "strawberry plants with red ripe fruits hanging from a raised bed in a plastic greenhouse", "strawberry plant in a grow bag with coco peat substrate, yellow sticky trap",
          "young strawberry runners and leaves on a white plastic tray, pebbles on the floor", "indoor strawberry vertical farm under white LED lights, metal frames"]
PHOTO = ["photo, realistic, smartphone photo", "DSLR photo, natural light, high detail", "realistic photograph, 35mm", "phone camera snapshot, slightly overexposed"]
STYLES = ["watercolor painting", "digital illustration, clean lines", "oil painting", "anime style background art", "colored pencil drawing",
          "vector art, flat colors", "impressionist painting", "studio ghibli style illustration", "3d render, cartoon style", "pastel illustration"]
NEG = "text, watermark, blurry, lowres, deformed"

def pipes(kind):
    from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
    if kind in ("ai_gen", "full_style_sd", "partial_style_sd"):
        base = StableDiffusionPipeline.from_pretrained(LOCAL / "weights/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", safety_checker=None)
        p = base if kind == "ai_gen" else StableDiffusionImg2ImgPipeline(**base.components)
    else:
        p = StableDiffusionInpaintPipeline.from_pretrained(LOCAL / "weights/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16", safety_checker=None)
    p.enable_attention_slicing(); p.enable_vae_slicing()
    p.enable_model_cpu_offload()
    p.set_progress_bar_config(disable=True)
    return p

def fit512(img, long=768):
    w, h = img.size; s = long / max(w, h)
    return img.resize((int(w * s) // 8 * 8, int(h * s) // 8 * 8), Image.LANCZOS)

def main(kind, n, start=0):
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "img").mkdir(exist_ok=True)
    man = [r for r in csv.DictReader(open(RAW / "manifest.csv", encoding="utf-8")) if r["tar"] != "VS_2.설향1"]
    rng = random.Random(zlib.crc32(f"im07|{kind}".encode()))
    rng.shuffle(man)
    p = pipes(kind)
    mp = OUT / "manifest.csv"; new = not mp.exists()
    f = open(mp, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["file", "label", "kind", "info", "source", "group", "farm_code", "tar", "other"])
    if new: w.writeheader()
    t = time.time()
    for i in range(start, start + n):
        r = man[i % len(man)]; lr = random.Random(zlib.crc32(f"{kind}|{i}".encode()))
        g = torch.Generator("cpu").manual_seed(lr.randint(0, 2**31))
        steps = lr.randint(20, 30); cfg = lr.uniform(5, 9)
        if kind == "ai_gen":
            prompt = f"{lr.choice(SCENES)}, {lr.choice(PHOTO)}"
            port = lr.random() < .6; W, H = (576, 768) if port else (768, 576)
            out = p(prompt, negative_prompt=NEG, width=W, height=H, num_inference_steps=steps, guidance_scale=cfg, generator=g).images[0]
            src = ""; info = f"{prompt}|{steps}|{cfg:.1f}"; r = dict(group=f"gen{i}", farm_code="GEN", tar="GEN")
        else:
            src = r["file"]; img = Image.open(RAW / src).convert("RGB")
            base = fit512(dist_resize(img, lr)); a = np.asarray(base)
            if kind in ("full_style_sd", "partial_style_sd"):
                stl = lr.choice(STYLES); s = lr.uniform(.45, .75)
                out = p(f"strawberry plants, {stl}", image=base, strength=s, num_inference_steps=steps, guidance_scale=cfg, negative_prompt=NEG, generator=g).images[0]
                out = out.resize(base.size, Image.LANCZOS); info = f"{stl}|s={s:.2f}"
                if kind == "partial_style_sd":
                    m = plant_mask(a) * blob_mask(a.shape, lr, (.15, .5)) if lr.random() < .6 else blob_mask(a.shape, lr)
                    if m.sum() < .03 * m.size: m = blob_mask(a.shape, lr)
                    fm = feather(m, lr.randint(2, 10))[..., None]
                    out = Image.fromarray((np.asarray(out) * fm + a * (1 - fm)).astype(np.uint8))
            else:  # partial_gen_sd
                m = blob_mask(a.shape, lr, (.04, .2))
                mask = Image.fromarray(m * 255)
                prompt = lr.choice(["strawberry leaves", "strawberry flower", "ripe strawberry fruit", "green leaves", "white plastic gutter",
                                    "pebbles", "drip irrigation tube", "yellow sticky trap", "coco peat substrate", "strawberry plant"]) + ", photo, realistic"
                gen = p(prompt, image=base, mask_image=mask, width=base.size[0], height=base.size[1], num_inference_steps=steps, guidance_scale=cfg, generator=g).images[0]
                gen = gen.resize(base.size, Image.LANCZOS)
                keep_whole = lr.random() < .3   # 일부는 VAE 재구성 전체 이미지 그대로(생성기 전체 흔적)
                if keep_whole: out = gen
                else:
                    fm = feather((m > 0).astype(np.uint8), lr.randint(1, 8))[..., None]
                    out = Image.fromarray((np.asarray(gen) * fm + a * (1 - fm)).astype(np.uint8))
                info = f"{prompt}|area={m.mean():.3f}|whole={int(keep_whole)}"
        final = dist_resize(out, lr)
        fn = f"{kind}_{i:05d}.jpg"; (OUT / "img" / fn).write_bytes(jpeg_bytes(final))
        w.writerow(dict(file=fn, label=1, kind=kind, info=info, source=src, group=r["group"], farm_code=r["farm_code"], tar=r["tar"], other=""))
        f.flush()
        print(f"{kind} {i} {time.time()-t:.0f}s", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
