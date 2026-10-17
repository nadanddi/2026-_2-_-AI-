"""팜모니 이미지 미션 제출 09 — 나단8 + GPT 재그리기 전문가 E_gr(저수준 38 + L/14 조각평균 임베딩 로지스틱, model/gr_expert.npz). inference.py
 E_gr은 6)의 다른 전문가와 같은 방식(θ·θ_rc, 형식 게이트)으로 p ← max(p, g(s_gr, θ)).
(이하 제출 08 설명) 팜모니 이미지 미션 제출 08 — 나단7(=나단6 출력) + 유형별 전문가 H(필터 E_f, 부분 스타일 조각 E_ps2, 국소 재생성 조각 E_g7).
 6) 전문가 H(고정 규칙, 정밀 원본 대조 뒤): s_f = 로지스틱(저수준 38 + L/14 조각평균 임베딩), s_ps2·s_g7 = B/16 54조각 로지스틱 최댓값.
    θ = 진짜 확정이 발동한 사진이면 theta_rc, 아니면 theta(model/config.json "h"). p ← max(p, g(s, θ)). 전문가 실패 → 미적용.
    형식 게이트: 파일이 배포 형식(JPEG, 크로마 4:2:0, Q92 양자화표)일 때만 6)을 적용하고, 아니면 제출 06/07과 같은 출력.
(이하 제출 07 설명)
 5) 정밀 원본 대조(고정 규칙): 사진의 정밀 지문(흑백 24×32 + 64px 구역 라플라시안 분산 log)을 AI Hub 원천 152,043장 정밀 지문 DB(model/finefp_db.npz)와 L1 대조.
    1위 거리 dg1 ≤ m 이고 1·2위 차 gap ≥ gap_min 이면 원본 있음. 원본 있음 + 국소 차이(loc_g > f_g 또는 loc_e > f_e) → p ← max(p, 0.95)(가짜 확정).
    원본 있음 + 국소·전체 차이 모두 작음(loc_g ≤ r_g, loc_e ≤ r_e, |glob_e| ≤ r_glob) + p < 0.8 → p ← min(p, 0.3)(진짜 확정). 값은 model/config.json "finefp".
(이하 제출 04·05와 같은 설명)
제출 03(나단3)과 모델·문턱·결합 규칙이 같고, 계산 순서만 바꿨습니다(채점 5분 한도 초과 반려 대응).
 - 이미지 묶음 계산: 경로를 정렬한 뒤 8장씩 묶어 CLIP(L/14 9조각, B/16 54조각)을 한 번에 계산 → GPU 활용 향상.
   정렬 후 묶으므로 입력 순서가 바뀌어도 묶음 구성과 결과가 같습니다.
 - CPU(저수준 38특징)와 GPU 계산을 동시에 진행(스레드).
 - 원본 대조 지문 DB를 GPU에 float로 한 번만 올려 두고 묶음 단위 L1 거리 계산.
최종 출력 처리(README 4절과 동일, 고정 규칙):
 1) 기본: p1 = 로지스틱(ViT-L/14@336 3×3 조각 평균), p2 = 부스팅(저수준 38특징), p = (p1+p2)/2
 2) R1: 3.5 < d1 < 20 이고 gap > 8 이고 p ≥ 0.25 이면 p ← max(p, 0.9)
 3) 전문가 E_pg·E_ps(같은 임베딩 로지스틱), E_L(ViT-B/16 54조각 로지스틱 최댓값), 4) p ← max(p, g_k(s_k))
 예외 대체(고정): 저수준 실패 → p1, CLIP 실패 → p2, 둘 다 실패 → 0.5, 원본 대조 실패 → 규칙 미적용, 전문가 실패 → 미적용, NaN → 0.5.
네트워크 접근·다운로드·설치 없음. GPU가 없으면 CPU로 동작.
"""
import json, os, io, lzma
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from lowfeat import low_features  # noqa: E402
from vit import ViTL14_336  # noqa: E402


def _std_format(path):   # 배포 형식 판별: JPEG + 4:2:0 + PIL Q92 양자화표(공개샘플 21장 실측과 같음)
    try:
        from PIL import JpegImagePlugin
        b = io.BytesIO(); Image.new("RGB", (64, 64), (120, 130, 140)).save(b, "JPEG", quality=92); ref = Image.open(io.BytesIO(b.getvalue())).quantization
        im = Image.open(path); return im.format == "JPEG" and JpegImagePlugin.get_sampling(im) == 2 and im.quantization == ref
    except Exception:
        return False


def _load_xz(path):   # model/*.npz.xz = lzma로 압축한 npz(값은 압축 전과 동일)
    with lzma.open(path, "rb") as f:
        return np.load(io.BytesIO(f.read()))

MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
S, PS, PST, BATCH = 336, 224, 112, 8


def _grid(w, h):
    return [(int(x), int(y)) for y in np.linspace(0, h - S, 3).astype(int) for x in np.linspace(0, w - S, 3).astype(int)]


def _patch_grid(w, h):
    xs = np.linspace(0, w - PS, int(np.ceil((w - PS) / PST)) + 1).round().astype(int)
    ys = np.linspace(0, h - PS, int(np.ceil((h - PS) / PST)) + 1).round().astype(int)
    return [(int(x), int(y)) for y in ys for x in xs]


def _fingerprint(img):
    w, h = img.size
    port = h >= w
    tw, th = (768, 1024) if port else (1024, 768)
    s = max(tw / w, th / h)
    nw, nh = w * s, h * s
    x0, y0 = (nw - tw) / 2 / s, (nh - th) / 2 / s
    c = img.crop((round(x0), round(y0), round(x0 + tw / s), round(y0 + th / s)))
    fp = np.asarray(c.convert("RGB").resize((12, 16) if port else (16, 12), Image.BOX), np.uint8)
    return (fp if port else fp.transpose(1, 0, 2)).astype(np.float32).reshape(-1), port


def _hgb(ex, X):
    raw = np.full(len(X), ex["base"][0])
    off = 0
    for L in ex["lens"]:
        t = ex["nodes"][off:off + L]
        off += L
        node = np.zeros(len(X), int)
        for _ in range(64):
            leaf = t[node, 4] > 0
            if leaf.all():
                break
            f = t[node, 0].astype(int)
            go_left = X[np.arange(len(X)), f] <= t[node, 1]
            nxt = np.where(go_left, t[node, 2], t[node, 3]).astype(int)
            node = np.where(leaf, node, nxt)
        raw += t[node, 5]
    return 1.0 / (1.0 + np.exp(-raw))


def _lr(m, X):
    z = ((X - m["mean"]) / m["scale"]) @ m["coef"] + m["intercept"][0]
    return 1.0 / (1.0 + np.exp(-z))


def _g(s, th):
    return 0.5 * s / th if s < th else 0.5 + 0.5 * (s - th) / (1.0 - th)


def load_model(model_dir=None):
    from safetensors.torch import load_file
    model_dir = Path(model_dir) if model_dir else HERE / "model"
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float16 if dev.type == "cuda" else torch.float32
    visual = ViTL14_336()
    visual.load_state_dict({k: v.float() for k, v in load_file(str(model_dir / "b2_visual_fp16.safetensors")).items()})
    b16 = ViTL14_336(d=768, layers=12, heads=12, patch=16, image=224, out=512)
    b16.load_state_dict({k: v.float() for k, v in load_file(str(model_dir / "b16_visual_fp16.safetensors")).items()})
    visual, b16 = visual.eval().to(dev, dt), b16.eval().to(dev, dt)
    db = _load_xz(model_dir / "fp_db.npz.xz")
    fp, port = db["fp"].reshape(len(db["fp"]), -1), db["portrait"].astype(bool)
    dbt = {p: torch.from_numpy(fp[port == p].astype(np.float32)).to(dev) for p in (True, False)}   # 한 번만 float로 올림
    ld = lambda n: dict(np.load(model_dir / n))
    ff = _load_xz(model_dir / "finefp_db.npz.xz")
    fdg = torch.from_numpy(ff["g"].reshape(len(ff["g"]), -1).astype(np.float32)).to(dev); fde = ff["e"].reshape(len(ff["e"]), -1).astype(np.float32)
    hz = np.load(model_dir / "h_experts.npz"); hx = {k: {a: hz[f"{k}__{a}"] for a in ("mean", "scale", "coef", "intercept")} for k in ("E_f", "E_ps2", "E_g7")}
    gz = np.load(model_dir / "gr_expert.npz"); hx["E_gr"] = {a: gz[f"E_gr__{a}"] for a in ("mean", "scale", "coef", "intercept")}
    return dict(visual=visual, b16=b16, dev=dev, dt=dt, lr=ld("b2_lr.npz"), hg=ld("low_hgb.npz"), db=dbt, cfg=cfg,
                e_pg=ld("e_pg_lr.npz"), e_ps=ld("e_ps_lr.npz"), e_l=ld("e_l_lr.npz"), fdg=fdg, fde=fde, hx=hx)


def _finefp(img):   # 정밀 지문(학습 때 DB와 같은 계산; 배포 크기면 축소·크롭은 그대로 통과)
    im = img.convert("RGB"); w, h = im.size; port = h >= w; tw, th = (768, 1024) if port else (1024, 768); s = max(tw / w, th / h)
    nw, nh = max(tw, round(w * s)), max(th, round(h * s))
    if (nw, nh) != (w, h): im = im.resize((nw, nh), Image.LANCZOS)
    x0, y0 = (nw - tw) // 2, (nh - th) // 2; a = np.asarray(im.crop((x0, y0, x0 + tw, y0 + th)).convert("L"), np.float32)
    if not port: a = np.ascontiguousarray(np.rot90(a))
    g = cv2.resize(a, (24, 32), interpolation=cv2.INTER_AREA); lap = cv2.Laplacian(a, cv2.CV_32F, ksize=3); v = lap.reshape(16, 64, 12, 64).var(axis=(1, 3))
    e = np.clip(np.round((np.log1p(v) / 10.0) * 255), 0, 255)
    return np.clip(np.round(g), 0, 255).astype(np.float32).ravel(), e.astype(np.float32).ravel()


def _norm(x, m):
    return ((x.to(m["dev"]).float() / 255.0 - MEAN.to(m["dev"])) / STD.to(m["dev"])).to(m["dt"])


@torch.no_grad()
def _batch_gpu(m, imgs):
    """imgs: PIL 목록(None 허용) → 이미지별 (emb, top1_L, (d1,d2)) 또는 실패 시 None 항목."""
    out = [dict() for _ in imgs]
    ok = [i for i, im in enumerate(imgs) if im is not None]
    arrs = {i: np.asarray(imgs[i], np.uint8) for i in ok}
    try:   # L/14 9조각 × 묶음
        cr = [torch.from_numpy(arrs[i][y:y + S, x:x + S]).permute(2, 0, 1) for i in ok for x, y in _grid(imgs[i].size[0], imgs[i].size[1])]
        e = m["visual"](_norm(torch.stack(cr), m)).float().cpu().numpy().reshape(len(ok), 9, -1)
        e = e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-8)
        for j, i in enumerate(ok): out[i]["emb"] = e[j].mean(0)
    except Exception:
        pass
    try:   # B/16 54조각 × 묶음(108조각씩)
        pc, own = [], []
        for i in ok:
            h, w = arrs[i].shape[:2]
            for x, y in _patch_grid(w, h): pc.append(torch.from_numpy(arrs[i][y:y + PS, x:x + PS]).permute(2, 0, 1)); own.append(i)
        es = []
        for k in range(0, len(pc), 216):
            es.append(m["b16"](_norm(torch.stack(pc[k:k + 216]), m)).float().cpu().numpy())
        es = np.concatenate(es); es = es / (np.linalg.norm(es, axis=-1, keepdims=True) + 1e-8); sc = _lr(m["e_l"], es); own = np.array(own)
        for i in ok:
            out[i]["top1"] = float(sc[own == i].max())
            try:   # 6) 전문가 H 조각 점수(같은 B/16 임베딩)
                ei = es[own == i]; out[i]["s_ps2"] = float(_lr(m["hx"]["E_ps2"], ei).max()); out[i]["s_g7"] = float(_lr(m["hx"]["E_g7"], ei).max())
            except Exception:
                pass
    except Exception:
        pass
    try:   # 정밀 원본 대조(묶음)
        ge = {i: _finefp(imgs[i]) for i in ok}; qi = list(ge)
        q = torch.from_numpy(np.stack([ge[i][0] for i in qi])).to(m["dev"]); d = torch.cdist(q, m["fdg"], p=1) / q.shape[1]
        v, ix = torch.topk(d, 2, largest=False); v, ix = v.cpu().numpy(), ix.cpu().numpy()
        for j, i in enumerate(qi):
            i1 = ix[j, 0]; dg = np.abs(ge[i][0] - m["fdg"][i1].cpu().numpy()); de = ge[i][1] - m["fde"][i1]; c = np.abs(de - np.median(de))
            out[i]["ff"] = (float(v[j, 0]), float(v[j, 1] - v[j, 0]), float(np.sort(dg - np.median(dg))[-int(len(dg) * .02):].mean()),
                            float(np.sort(c)[-int(len(c) * .05):].mean()), float(np.median(de)))
    except Exception:
        pass
    for i in ok:   # 원본 대조
        try:
            fpv, port = _fingerprint(imgs[i])
            q = torch.from_numpy(fpv).to(m["dev"])[None]
            d = torch.cdist(q, m["db"][port], p=1)[0] / q.shape[1]
            two = torch.topk(d, 2, largest=False).values.cpu().numpy()
            out[i]["d"] = (float(two[0]), float(two[1]))
        except Exception:
            pass
    return out


def _safe_low(path):
    try:
        v = low_features(path)
        return v if np.all(np.isfinite(v)) else None
    except Exception:
        return None


def _final(m, info, low):
    rule, th = m["cfg"]["l2_rule"], m["cfg"]["experts"]["theta"]
    p1 = p2 = None
    try:
        if "emb" in info: p1 = float(_lr(m["lr"], info["emb"][None])[0])
    except Exception:
        p1 = None
    try:
        if low is not None: p2 = float(_hgb(m["hg"], low[None, :])[0])
    except Exception:
        p2 = None
    vals = [v for v in (p1, p2) if v is not None and np.isfinite(v)]
    p = float(np.mean(vals)) if vals else 0.5
    try:
        if "d" in info:
            d1, d2 = info["d"]
            if rule["d1_lo"] < d1 < rule["d1_hi"] and (d2 - d1) > rule["gap"] and p >= rule["min_p"]:
                p = max(p, rule["floor"])
    except Exception:
        pass
    p = float(np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0))
    for k, fn in (("E_pg", lambda: float(_lr(m["e_pg"], info["emb"][None])[0])), ("E_ps", lambda: float(_lr(m["e_ps"], info["emb"][None])[0])), ("E_L", lambda: info["top1"])):
        try:
            s = fn()
            if np.isfinite(s): p = max(p, _g(s, th[k]))
        except Exception:
            pass
    p = min(max(p, 0.0), 1.0)
    rc = False
    try:   # 5) 정밀 원본 대조
        if "ff" in info:
            T = m["cfg"]["finefp"]; dg1, gap, lg, le, ge_ = info["ff"]
            if dg1 <= T["m"] and gap >= T["gap"]:
                if lg > T["f_g"] or le > T["f_e"]: p = max(p, 0.95)
                elif lg <= T["r_g"] and le <= T["r_e"] and abs(ge_) <= T["r_glob"]: p = min(p, 0.3); rc = True
    except Exception:
        pass
    H = m["cfg"]["h"]; th = H["theta_rc"] if rc else H["theta"]
    for k, fn in () if not info.get("std", False) else (("E_f", lambda: float(_lr(m["hx"]["E_f"], np.r_[low, info["emb"]][None])[0])), ("E_ps2", lambda: info["s_ps2"]), ("E_g7", lambda: info["s_g7"]),
                                                     ("E_gr", lambda: float(_lr(m["hx"]["E_gr"], np.r_[low, info["emb"]][None])[0]))):
        try:   # 6) 전문가 H
            s = fn()
            if np.isfinite(s): p = max(p, _g(s, th[k]))
        except Exception:
            pass
    return min(max(p, 0.0), 1.0)


def predict(model, image_paths):
    paths = [str(p) for p in image_paths]
    order = sorted(range(len(paths)), key=lambda i: paths[i])   # 정렬 후 묶음 → 입력 순서와 무관
    res = [0.5] * len(paths)
    workers = max(1, min(8, os.cpu_count() or 1))
    with ThreadPoolExecutor(workers) as ex:
        lows = {i: ex.submit(_safe_low, paths[i]) for i in order}   # CPU 계산을 먼저 걸어 두고 GPU와 동시 진행
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]; imgs = []
            for i in idx:
                try: imgs.append(Image.open(paths[i]).convert("RGB"))
                except Exception: imgs.append(None)
            infos = _batch_gpu(model, imgs)
            for i, info in zip(idx, infos):
                info["std"] = _std_format(paths[i])
                res[i] = _final(model, info, lows[i].result())
    return res


if __name__ == "__main__":
    ps = sorted(str(p) for p in Path(sys.argv[1]).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    mm = load_model()
    for p, pr in zip(ps, predict(mm, ps)):
        print(f"{Path(p).name}  {pr:.4f}")
