"""팜모니 이미지 미션 제출 03(C3: 유형별 전문가 위원회) — inference.py
채점기는 이 파일을 import 한 뒤 load_model(model_dir) / predict(model, image_paths) 두 함수만 호출합니다.

최종 출력 처리 (README 4절과 동일, 고정 규칙 · 실행 중 갱신 없음)
 1) 기본 판별기(제출 01과 같은 모델): p1 = 로지스틱(CLIP ViT-L/14@336 3×3 조각 평균 임베딩), p2 = 결정나무 부스팅(저수준 38특징), p = (p1+p2)/2
 2) 원본 대조 규칙(R1): 3.5 < d1 < 20 이고 gap > 8 이고 p ≥ 0.25 이면 p ← max(p, 0.9)
 3) 전문가 셋(각자 고정 문턱 θ_k)
    - E_pg(부분 생성 전문)·E_ps(부분 스타일 전문): 1)과 같은 조각 평균 임베딩 → 각자의 로지스틱 → s_k
    - E_L(국소 조각 전문): CLIP ViT-B/16로 224×224 조각 54개(간격 112) 임베딩 → 조각 로지스틱 → 최댓값 s_L
 4) 최종 p ← max(p, g_k(s_k)),  g_k(s) = 0.5·s/θ_k (s < θ_k) 또는 0.5 + 0.5·(s−θ_k)/(1−θ_k)
    → 기본이 0.5 이상이거나 전문가 하나라도 자기 문턱 이상이면 0.5 이상
 예외 대체(고정): 저수준 실패 → p1, CLIP 실패 → p2, 둘 다 실패 → 0.5, 원본 대조 실패 → 규칙 미적용, 전문가 계산 실패 → 그 전문가 미적용, NaN → 0.5.
이미지는 한 장씩 독립 처리하므로 입력 순서가 바뀌어도 결과가 같습니다. 네트워크 접근·다운로드·설치 없음. GPU가 없으면 CPU로 동작.
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

MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
S = 336
PS, PST = 224, 112


def _grid_crops(img):
    w, h = img.size
    out = []
    for yy in np.linspace(0, h - S, 3).astype(int):
        for xx in np.linspace(0, w - S, 3).astype(int):
            out.append(img.crop((int(xx), int(yy), int(xx) + S, int(yy) + S)))
    return out


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


def load_model(model_dir=None):
    from safetensors.torch import load_file
    model_dir = Path(model_dir) if model_dir else HERE / "model"
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual = ViTL14_336()
    visual.load_state_dict({k: v.float() for k, v in load_file(str(model_dir / "b2_visual_fp16.safetensors")).items()})
    visual = visual.eval().to(dev)
    b16 = ViTL14_336(d=768, layers=12, heads=12, patch=16, image=224, out=512)   # 같은 구조 클래스, ViT-B/16 크기
    b16.load_state_dict({k: v.float() for k, v in load_file(str(model_dir / "b16_visual_fp16.safetensors")).items()})
    b16 = b16.eval().to(dev)
    if dev.type == "cuda":
        visual, b16 = visual.half(), b16.half()
    db = np.load(model_dir / "fp_db.npz")
    fp, port = db["fp"].reshape(len(db["fp"]), -1), db["portrait"].astype(bool)
    dbt = {p: torch.from_numpy(fp[port == p]).to(dev) for p in (True, False)}
    ld = lambda n: dict(np.load(model_dir / n))
    return dict(visual=visual, b16=b16, dev=dev, lr=ld("b2_lr.npz"), hg=ld("low_hgb.npz"), db=dbt, cfg=cfg,
                e_pg=ld("e_pg_lr.npz"), e_ps=ld("e_ps_lr.npz"), e_l=ld("e_l_lr.npz"))


@torch.no_grad()
def _embed(m, img):
    xs = [(torch.from_numpy(np.asarray(c, np.float32) / 255.0).permute(2, 0, 1) - MEAN) / STD for c in _grid_crops(img)]
    x = torch.stack(xs).to(m["dev"])
    if m["dev"].type == "cuda":
        x = x.half()
    e = m["visual"](x).float().cpu().numpy()
    e = e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-8)
    return e.mean(0)


@torch.no_grad()
def _patch_top1(m, img):
    a = np.asarray(img, np.float32) / 255.0
    h, w = a.shape[:2]
    xs = [(torch.from_numpy(a[y:y + PS, x:x + PS]).permute(2, 0, 1) - MEAN) / STD for x, y in _patch_grid(w, h)]
    x = torch.stack(xs).to(m["dev"])
    if m["dev"].type == "cuda":
        x = x.half()
    e = m["b16"](x).float().cpu().numpy()
    e = e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-8)
    return float(_lr(m["e_l"], e).max())


@torch.no_grad()
def _lookup(m, fpv, port):
    D = m["db"][port]
    q = torch.from_numpy(fpv).to(m["dev"])
    d = torch.empty(D.shape[0], device=m["dev"])
    for i in range(0, D.shape[0], 32768):
        d[i:i + 32768] = (D[i:i + 32768].float() - q).abs().mean(1)
    two = torch.topk(d, 2, largest=False).values.cpu().numpy()
    return float(two[0]), float(two[1])


def _g(s, th):
    return 0.5 * s / th if s < th else 0.5 + 0.5 * (s - th) / (1.0 - th)


@torch.no_grad()
def _one(m, path, low):
    rule, ex = m["cfg"]["l2_rule"], m["cfg"]["experts"]["theta"]
    p1 = p2 = None
    img = emb = None
    try:
        img = Image.open(path).convert("RGB")
        emb = _embed(m, img)          # 항상 이미지 1장씩 계산 → 순서·묶음과 무관
        p1 = float(_lr(m["lr"], emb[None])[0])
    except Exception:
        p1 = None
    try:
        if low is not None:
            p2 = float(_hgb(m["hg"], low[None, :])[0])
    except Exception:
        p2 = None
    vals = [v for v in (p1, p2) if v is not None and np.isfinite(v)]
    p = float(np.mean(vals)) if vals else 0.5
    try:
        if img is not None:
            fpv, port = _fingerprint(img)
            d1, d2 = _lookup(m, fpv, port)
            if rule["d1_lo"] < d1 < rule["d1_hi"] and (d2 - d1) > rule["gap"] and p >= rule["min_p"]:
                p = max(p, rule["floor"])
    except Exception:
        pass
    p = float(np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0))
    for k, fn in (("E_pg", lambda: float(_lr(m["e_pg"], emb[None])[0])), ("E_ps", lambda: float(_lr(m["e_ps"], emb[None])[0])), ("E_L", lambda: _patch_top1(m, img))):
        try:
            if img is None or (emb is None and k != "E_L"):
                continue
            s = fn()
            if np.isfinite(s):
                p = max(p, _g(s, ex[k]))
        except Exception:
            pass
    return min(max(p, 0.0), 1.0)


def _safe_low(path):
    try:
        v = low_features(path)
        return v if np.all(np.isfinite(v)) else None
    except Exception:
        return None


def predict(model, image_paths):
    workers = max(1, min(8, os.cpu_count() or 1))
    with ThreadPoolExecutor(workers) as ex:
        lows = list(ex.map(_safe_low, image_paths))
    return [_one(model, p, lw) for p, lw in zip(image_paths, lows)]


if __name__ == "__main__":
    paths = sorted(str(p) for p in Path(sys.argv[1]).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    mm = load_model()
    for p, pr in zip(paths, predict(mm, paths)):
        print(f"{Path(p).name}  {pr:.4f}")
