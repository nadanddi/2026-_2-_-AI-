"""IM31: Qwen-Image-Edit(Apache-2.0, 연구실 서버 GPU 0)로 지시문 편집 가짜 제작 (IM_필요이미지_정리_v1.md 1·2·4번). 환경 nadan_qwen.
입력: local/im_new_raw/<tar>/<file> 중 role=gen (manifest.csv). 원본은 배포 형식(덮게 축소·크롭)으로 맞춘 뒤 편집.
종류(원본마다 하나, crc32로 고정 배정):
  pgen  부분 생성(추가·지우기·바꾸기, 크기 '잎 몇 장'~'포기 하나')
  pstyle 일부만 화풍 변환 / pcolor 일부 잎 색만 변경 / brown 일부 잎 갈변·누렇게
  regen 같은 장면 사실적 재그리기(편집 없음) / fstyle_photo 구도 유지·조명·색감만 다른 사진풍 재그리기
출력: local/im_set_qwen/raw/<id>.png(받은 그대로), manifest_qwen.csv(id, kind, prompt, seed, source, farm_code, group) — 재개 가능
사용: python im31_qwen_edit.py <n> [steps=8]  (Lightning LoRA 8단계, cfg 1, 영어 지시문 + 원본 사진 느낌 유지 문구; 시험 결과 "smartphone" 단어는 카메라 UI를 그려 뺌)
"""
import sys, csv, random, zlib, time, os
from pathlib import Path
import torch
from PIL import Image
LOCAL = Path(__file__).resolve().parents[1] / "local"; W = LOCAL / "weights/qwen-image-edit"; OUT = LOCAL / "im_set_qwen"
LORA = LOCAL / "weights/qwen-lightning/Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"   # Apache-2.0, 8단계
POS = ["upper-left", "upper-right", "center", "lower-left", "lower-right", "front"]
SIZE = ["a small area of two or three leaves", "an area about the size of one plant"]
ADD = ["a few white strawberry flowers", "one ripe red strawberry", "a few small green unripe strawberries", "some new leaves", "a strawberry runner stem"]
REMOVE = ["the plant label", "one black irrigation tube", "the yellow sticky trap", "a withered leaf", "a small strawberry plant"]
STYLE = ["watercolor painting", "oil painting", "colored pencil drawing", "digital painting", "pastel drawing"]
KINDS = [("pgen", 40), ("pstyle", 15), ("pcolor", 10), ("brown", 10), ("regen", 15), ("fstyle_photo", 10)]
KEEP = (" Keep everything else exactly the same. Preserve the original photo's exact colors, white balance, lighting, sharpness, grain and noise."
        " Do not beautify, do not increase saturation. It must look like an unedited real photograph. No text, no UI, no borders, no people.")

def prompt_for(kind, r):
    pos, size = r.choice(POS), r.choice(SIZE)
    if kind == "pgen":
        a = r.choice(["add", "remove", "replace"])
        if a == "add": return f"Add {r.choice(ADD)} naturally in {size} in the {pos} part of the photo." + KEEP
        if a == "remove": return f"Remove {r.choice(REMOVE)} in the {pos} part of the photo and fill the area naturally." + KEEP
        return f"Replace the strawberry plant in the {pos} part of the photo with a similar but different strawberry plant." + KEEP
    if kind == "pstyle": return f"Change only {size} in the {pos} part of the photo into a {r.choice(STYLE)} style, {r.choice(['subtly', 'clearly'])}; keep the rest of the photo unchanged." + KEEP.replace(" Keep everything else exactly the same.", "")
    if kind == "pcolor": return f"Change only the color of the leaves in {size} in the {pos} part of the photo to {r.choice(['slightly more yellowish green', 'a bluish green tint', 'a desaturated grayish green', 'a deeper dark green'])}; keep texture and shape." + KEEP
    if kind == "brown": return f"Change only the leaves in {size} in the {pos} part of the photo into {r.choice(['leaves with brown dried edges', 'yellowed leaves', 'brown withered leaves'])}; keep their shape and position." + KEEP
    if kind == "regen": return "Re-render the same scene photorealistically with slightly different natural lighting and camera characteristics; keep the composition and every object in place." + KEEP.replace(" Keep everything else exactly the same.", "")
    return f"Re-render this photo as if taken under {r.choice(['an overcast sky', 'early morning sunlight', 'late afternoon sunlight', 'fluorescent greenhouse lights'])}; keep the composition and objects, photorealistic." + KEEP.replace(" Keep everything else exactly the same.", "")

def dist_cover(img):
    w, h = img.size; tw, th = (768, 1024) if h >= w else (1024, 768); s = max(tw / w, th / h)
    r = img.resize((max(tw, round(w * s)), max(th, round(h * s))), Image.LANCZOS); x0, y0 = (r.width - tw) // 2, (r.height - th) // 2
    return r.crop((x0, y0, x0 + tw, y0 + th))

def main(n, steps=8):
    from diffusers import QwenImageEditPipeline
    rows = [r for r in csv.DictReader(open(LOCAL / "im_new_raw/manifest.csv", encoding="utf-8")) if r["role"] == "gen"]
    (OUT / "raw").mkdir(parents=True, exist_ok=True); mp = OUT / "manifest_qwen.csv"
    done = {r["id"] for r in csv.DictReader(open(mp, encoding="utf-8"))} if mp.exists() else set()
    f = open(mp, "a", newline="", encoding="utf-8"); w = csv.DictWriter(f, fieldnames=["id", "kind", "prompt", "seed", "source", "tar", "farm_code", "group", "sec"])
    if f.tell() == 0: w.writeheader()
    pipe = QwenImageEditPipeline.from_pretrained(str(W), torch_dtype=torch.bfloat16); pipe.load_lora_weights(str(LORA)); pipe.fuse_lora()
    pipe.enable_model_cpu_offload(); pipe.set_progress_bar_config(disable=True)
    bag = [k for k, c in KINDS for _ in range(c)]; t0 = time.time(); k = 0
    for i, r in enumerate(rows[:n]):
        rid = f"q{i:05d}"
        if rid in done: continue
        rr = random.Random(zlib.crc32(("qwen|" + r["file"]).encode())); kind = rr.choice(bag); prompt = prompt_for(kind, rr); seed = rr.randint(0, 2**31 - 1)
        img = dist_cover(Image.open(LOCAL / "im_new_raw" / r["tar"] / r["file"]).convert("RGB")); ts = time.time()
        try:
            out = pipe(image=img, prompt=prompt, negative_prompt=" ", true_cfg_scale=1.0, num_inference_steps=steps, height=img.size[1], width=img.size[0], generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        except Exception as e:
            print("FAIL", rid, repr(e)[:200], flush=True); continue
        out.save(OUT / "raw" / f"{rid}.png")
        w.writerow(dict(id=rid, kind=kind, prompt=prompt, seed=seed, source=r["file"], tar=r["tar"], farm_code=r["farm_code"], group=r["group"], sec=round(time.time() - ts, 1))); f.flush(); k += 1
        if k % 10 == 0: print(f"{k}장 {time.time() - t0:.0f}s (장당 {(time.time() - t0) / k:.1f}s)", flush=True)
    print("QWEN_GEN_DONE", k, flush=True)

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 8)
