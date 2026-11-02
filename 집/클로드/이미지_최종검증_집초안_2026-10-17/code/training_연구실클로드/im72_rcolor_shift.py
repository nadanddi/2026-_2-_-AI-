"""IM72: 팀원 제출(나단9_김우진업데이트, R_color 규칙) 처리 내성 점검 — im70과 같은 R5 진짜 원본 변형(자르기 이동 등)에서 R_color 발동(진짜→가짜 경보) 비율. 서버(nadan, GPU 0). 판정 아님(팀원 모델 위험 확인).
팀원 inference.py·config.json + 서버 나단9 model 파일(심볼릭 링크)로 실제 코드 그대로 계산."""
import sys, io, csv, numpy as np
from pathlib import Path
from PIL import Image
PK = Path.home() / "storage/nadan/farmai_img/data/local/kwj_test/pkg"; sys.path.insert(0, str(PK)); import inference as I  # noqa
L = Path.home() / "storage/nadan/farmai_img/data/local"; m = I.load_model(str(PK / "model")); C = m["cfg"]["r_color"]; print("r_color", C, flush=True)
rr = [r for r in csv.DictReader(open(L / "im_set_r5/manifest.csv", encoding="utf-8")) if r["label"] == "0"]
def make(raw, kern=Image.LANCZOS, scale=1.0, sh=0, gamma=1.0, bright=1.0, q=92):
    w, h = raw.size; tw, th = (768, 1024) if h >= w else (1024, 768); s = max(tw / w, th / h) * scale
    nw, nh = max(tw, round(w * s)), max(th, round(h * s)); r = raw.resize((nw, nh), kern)
    x0 = int(np.clip((nw - tw) // 2 + sh, 0, nw - tw)); y0 = int(np.clip((nh - th) // 2 + sh, 0, nh - th)); a = np.asarray(r.crop((x0, y0, x0 + tw, y0 + th)), np.float32)
    if gamma != 1.0: a = 255 * (a / 255) ** gamma
    a = np.clip(a * bright, 0, 255).astype(np.uint8); b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG", quality=q, subsampling=2); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
raws = [Image.open(L / "im_r5_raw" / r["tar"] / r["source"]).convert("RGB") for r in rr]; print("R5 진짜", len(raws), flush=True)
VAR = [("기준", {}), ("이동2", dict(sh=2)), ("이동4", dict(sh=4)), ("이동8", dict(sh=8)), ("이동-2", dict(sh=-2)), ("이동-4", dict(sh=-4)), ("확대1.03", dict(scale=1.03)),
       ("감마0.9", dict(gamma=.9)), ("밝기+5%", dict(bright=1.05)), ("bicubic", dict(kern=Image.BICUBIC)), ("bilinear", dict(kern=Image.BILINEAR))]
for nm, kw in VAR:
    ims = [make(im, **kw) for im in raws]; infos = []
    for i in range(0, len(ims), 8): infos += I._batch_gpu(m, ims[i:i + 8])
    fire, dg, cl = 0, [], []
    for f in infos:
        if "ff" in f and "d" in f and "cl" in f:
            d1 = f["ff"][0]; cg = f["d"][1] - f["d"][0]; dg.append(d1); cl.append(f["cl"])
            fire += int(C["dg1_lo"] < d1 <= C["dg1_hi"] and cg > C["gap"] and f["cl"] > C["t"])
    print(f"[{nm:8s}] R_color 발동(진짜→가짜) {fire}/{len(ims)} = {fire / len(ims):.4f} | dg1 중앙 {np.median(dg):.2f} | cl 중앙 {np.median(cl):.2f} q99 {np.quantile(cl, .99):.2f}", flush=True)
print("IM72_DONE", flush=True)
