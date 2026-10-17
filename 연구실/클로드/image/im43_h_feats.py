"""IM43: 후보 H 특징 — 나단6 패키지 코드(submit_img06/inference.py)를 그대로 써서 사진마다 emb(768)·B/16 54조각 임베딩(54×512)·저수준 38·나단6 최종 p 저장.
서버(nadan, GPU 0). 사용: python im43_h_feats.py <세트이름> [...]   세트: im_set_* 또는 sealed_<real|brown|bright|qwen|v7> 또는 pub21 또는 jig500
출력: local/feats_h/<세트>.npz (files, emb, es(float16), low, p6, ok, ppre=정밀 대조 전 p, rc=진짜 확정 발동, ff)
"""
import sys, csv, numpy as np, torch
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
B = Path.home() / "storage/nadan/farmai_img"; L = B / "data/local"; PK = B / "submit_img06/팜모니_나단6"
sys.path.insert(0, str(PK)); import inference as I  # noqa
OUT = L / "feats_h"; OUT.mkdir(exist_ok=True)

def paths_of(sn):
    if sn.startswith("im_set_"): return [str(L / sn / "img" / r["file"]) for r in csv.DictReader(open(L / sn / "manifest.csv", encoding="utf-8"))]
    if sn.startswith("sealed_"): d = L / "sealed_c3" / sn[7:]; return [str(d / "img" / r["file"]) for r in csv.DictReader(open(d / "manifest.csv", encoding="utf-8"))]
    if sn == "pub21":
        P = Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플"; return [str(P / "images" / r["file"]) for r in csv.DictReader(open(P / "sample_labels.csv"))]
    if sn == "jig500": return sorted(str(p) for p in (L / "preds/_jig_hold").glob("j*.jpg"))
    raise ValueError(sn)

@torch.no_grad()
def patch_emb(m, img):
    a = np.asarray(img, np.uint8); h, w = a.shape[:2]
    pc = [torch.from_numpy(a[y:y + I.PS, x:x + I.PS]).permute(2, 0, 1) for x, y in I._patch_grid(w, h)]
    e = m["b16"](I._norm(torch.stack(pc), m)).float().cpu().numpy(); return e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-8)

def run(sn, m):
    ps = paths_of(sn); n = len(ps); emb = np.zeros((n, 768), np.float32); es = np.zeros((n, 54, 512), np.float16); low = np.full((n, 38), np.nan, np.float32); p6 = np.full(n, .5); ok = np.zeros(n, bool); ppre = np.full(n, .5); rc = np.zeros(n, bool); ff = np.full((n, 5), np.nan)
    with ThreadPoolExecutor(8) as ex:
        lows = [ex.submit(I._safe_low, p) for p in ps]
        for b in range(0, n, I.BATCH):
            idx = list(range(b, min(n, b + I.BATCH))); imgs = [Image.open(ps[i]).convert("RGB") for i in idx]; infos = I._batch_gpu(m, imgs)
            for i, im, info in zip(idx, imgs, infos):
                lv = lows[i].result(); p6[i] = I._final(m, info, lv); ppre[i] = I._final(m, {kk: vv for kk, vv in info.items() if kk != "ff"}, lv)
                if "ff" in info:
                    T = m["cfg"]["finefp"]; dg1, gap, lg, le, ge_ = info["ff"]; ff[i] = info["ff"]
                    rc[i] = dg1 <= T["m"] and gap >= T["gap"] and not (lg > T["f_g"] or le > T["f_e"]) and lg <= T["r_g"] and le <= T["r_e"] and abs(ge_) <= T["r_glob"] and ppre[i] < .8
                if "emb" in info and lv is not None:
                    emb[i] = info["emb"]; low[i] = lv; e = patch_emb(m, im)
                    if len(e) == 54: es[i] = e; ok[i] = True
            if b % 400 == 0: print(sn, b, "/", n, flush=True)
    np.savez(OUT / f"{sn}.npz", files=np.array([Path(p).name for p in ps]), emb=emb, es=es, low=low, p6=p6, ok=ok, ppre=ppre, rc=rc, ff=ff); print(sn, "저장", n, "ok", ok.sum(), flush=True)

if __name__ == "__main__":
    m = I.load_model(str(PK / "model"))
    for sn in sys.argv[1:]:
        if (OUT / f"{sn}.npz").exists(): print(sn, "이미 있음"); continue
        run(sn, m)
    print("FEATS_DONE", flush=True)
