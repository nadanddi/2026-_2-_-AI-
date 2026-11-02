"""팜모니 이미지 미션 제출 04 — C3(유형별 전문가 위원회) 속도 개선판. inference.py
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
import json, os
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
    db = np.load(model_dir / "fp_db.npz")
    fp, port = db["fp"].reshape(len(db["fp"]), -1), db["portrait"].astype(bool)
    dbt = {p: torch.from_numpy(fp[port == p].astype(np.float32)).to(dev) for p in (True, False)}   # 한 번만 float로 올림
    ld = lambda n: dict(np.load(model_dir / n))
    return dict(visual=visual, b16=b16, dev=dev, dt=dt, lr=ld("b2_lr.npz"), hg=ld("low_hgb.npz"), db=dbt, cfg=cfg,
                e_pg=ld("e_pg_lr.npz"), e_ps=ld("e_ps_lr.npz"), e_l=ld("e_l_lr.npz"))


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
        for i in ok: out[i]["top1"] = float(sc[own == i].max())
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
                res[i] = _final(model, info, lows[i].result())
    return res


if __name__ == "__main__":
    ps = sorted(str(p) for p in Path(sys.argv[1]).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    mm = load_model()
    for p, pr in zip(ps, predict(mm, ps)):
        print(f"{Path(p).name}  {pr:.4f}")
