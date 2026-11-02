"""IM07c: 후보 B용 SD 부분 편집 가짜 v3 (중간 비평 제안 3).
im07b(v2)와 다른 점 = 마스크 모양 다양화 + 생성 흔적은 마스크 안에만:
 - 마스크 종류: object(작물 마스크의 연결요소 일부 = 잎·포기 모양) .35 / rect(회전 사각형, 사람이 박스로 지정한 편집) .2 /
                brush(굵은 붓질 폴리라인) .2 / blob(타원 덩어리, v2 방식) .25
 - 마스크 면적: 타일(512)의 5~45%. 페더 0~8px. 마스크 밖은 원본 픽셀 그대로(VAE 재구성 흔적이 마스크 밖에 남지 않게).
 - 패치 위치 힌트(tile=)는 manifest에 남기되, CNN 학습에서 타일 내부 우선 추출은 쓰지 않을 예정(비평 치명-1(b)).
kind 이름은 v2와 같음(partial_gen_sd / partial_style_sd), info에 mask=종류, 출력 local/im_set_sd3.
사용: python im07c_sd_partial_v3.py <rounds>   (라운드당 부분 생성 6 + 부분 스타일 4)
"""
import sys, csv, random, zlib, time, math
import numpy as np, cv2, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im07b_sd_fakes_v2 as B  # noqa
from im03_make_fakes import jpeg_bytes, feather, plant_mask, blob_mask, RAW, LOCAL  # noqa
from im03c_make_fakes_v3 import dist_cover  # noqa

OUT = LOCAL / "im_set_sd3"

def mask_of(ta, lr):
    kind = lr.choices(["object", "rect", "brush", "blob"], [.35, .2, .2, .25])[0]; m = np.zeros((512, 512), np.uint8)
    target = lr.uniform(.05, .45)
    if kind == "object":
        pm = plant_mask(ta); n, lab, st, _ = cv2.connectedComponentsWithStats(pm, 8)
        comps = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 400]
        lr.shuffle(comps)
        for i in comps:
            m[lab == i] = 1
            if m.mean() >= target: break
        if m.mean() > target * 1.4:   # 큰 포기 하나가 타일을 덮는 경우: 크기 맞춘 창과 교집합(경계는 작물 모양 유지)
            for _ in range(20):
                win = blob_mask((512, 512), lr, (target * .8, target * 1.2)); mm = m * win
                if .5 * target <= mm.mean() <= 1.4 * target: break
            m = mm
        if m.mean() < .03: kind = "blob"
    if kind == "rect":
        while m.mean() < target:
            cx, cy = lr.randint(40, 472), lr.randint(40, 472); w, h = lr.randint(60, 300), lr.randint(60, 300)
            box = cv2.boxPoints(((cx, cy), (w, h), lr.uniform(0, 90))).astype(np.int32); cv2.fillPoly(m, [box], 1)
    elif kind == "brush":
        while m.mean() < target:
            pts = [(lr.randint(0, 511), lr.randint(0, 511))]
            for _ in range(lr.randint(2, 6)):
                a = lr.uniform(0, 2 * math.pi); L = lr.randint(40, 160)
                pts.append((int(np.clip(pts[-1][0] + L * math.cos(a), 0, 511)), int(np.clip(pts[-1][1] + L * math.sin(a), 0, 511))))
            cv2.polylines(m, [np.array(pts, np.int32)], False, 1, thickness=lr.randint(20, 70))
    elif kind == "blob":
        m = blob_mask((512, 512), lr, (max(.05, target - .1), target))
    return m, kind

def one(kind, i, man):
    lr = random.Random(zlib.crc32(f"v3|{kind}|{i}".encode()))
    g = torch.Generator("cpu").manual_seed(lr.randint(0, 2**31)); steps = lr.randint(20, 30); cfg = lr.uniform(5, 9)
    P = B.get(kind)
    r = man[lr.randrange(len(man))]
    base = dist_cover(Image.open(RAW / r["file"]).convert("RGB"), lr); a = np.asarray(base).copy(); W, H = base.size
    x0, y0 = lr.randint(0, W - 512), lr.randint(0, H - 512); tile = base.crop((x0, y0, x0 + 512, y0 + 512)); ta = a[y0:y0 + 512, x0:x0 + 512]
    m, mk = mask_of(ta, lr)
    if kind == "partial_gen_sd":
        prompt = f"{lr.choice(B.INP)}, photo, realistic"
        gen = np.asarray(P["inp"](prompt, image=tile, mask_image=Image.fromarray(m * 255), width=512, height=512,
                                  num_inference_steps=steps, guidance_scale=cfg, generator=g).images[0])
    else:
        photo = lr.random() < .5; s = lr.uniform(.25, .45) if photo else lr.uniform(.4, .65)
        prompt = f"strawberry plants, {lr.choice(B.PHOTO_EDIT if photo else B.ART)}"
        gen = np.asarray(P["i2i"](prompt, image=tile, strength=s, num_inference_steps=steps, guidance_scale=cfg, negative_prompt=B.NEG, generator=g).images[0])
    fm = feather(m, lr.randint(0, 8))[..., None]
    a[y0:y0 + 512, x0:x0 + 512] = np.clip(gen * fm + ta * (1 - fm), 0, 255).astype(np.uint8)
    meta = dict(source=r["file"], group=r["group"], farm_code=r["farm_code"], tar=r["tar"])
    return Image.fromarray(a), meta, f"{prompt}|mask={mk}|area={m.mean():.3f}|tile={x0},{y0}"

def main(rounds):
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    red = set((LOCAL / "red_sources.txt").read_text(encoding="utf-8").split(chr(10))) - {""}
    man = [r for r in csv.DictReader(open(RAW / "manifest.csv", encoding="utf-8")) if r["file"] not in red]
    mp = OUT / "manifest.csv"; done = {r["file"] for r in csv.DictReader(open(mp, encoding="utf-8"))} if mp.exists() else set()
    f = open(mp, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["file", "label", "kind", "info", "source", "group", "farm_code", "tar", "other"])
    if f.tell() == 0: w.writeheader()
    t = time.time()
    for rd in range(rounds):
        for kind, k in [("partial_gen_sd", 6), ("partial_style_sd", 4)]:
            for j in range(k):
                i = rd * k + j; fn = f"{kind}_v3_{i:05d}.jpg"
                if fn in done: continue
                try: img, meta, info = one(kind, i, man)
                except Exception as e: print("FAIL", kind, i, repr(e)[:200], flush=True); torch.cuda.empty_cache(); continue
                (OUT / "img" / fn).write_bytes(jpeg_bytes(img)); w.writerow(dict(file=fn, label=1, kind=kind, info=info, other="", **meta)); f.flush()
        print(f"round {rd} {time.time()-t:.0f}s", flush=True)

if __name__ == "__main__":
    main(int(sys.argv[1]))
