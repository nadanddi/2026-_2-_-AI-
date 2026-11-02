"""팜모니 이미지 미션 제출 02(G3c1) — inference.py
제출 01과 다른 점: 학습 자료(GPT 생성·편집 250장 추가, 유형 안 10%)와 5) 고정 단조 변환 g(τ → 0.5). 계산 구조는 같음.
 5) p ← g(p): p < τ → 0.5·p/τ, 아니면 0.5 + 0.5·(p − τ)/(1 − τ). τ = model/config.json "tau"(학습 OOF 교차적합으로 미리 정한 값, 실행 중 갱신 없음).
채점기는 이 파일을 import 한 뒤 load_model(model_dir) / predict(model, image_paths) 두 함수만 호출합니다.

최종 출력 처리 (README 4절과 동일, 고정 규칙 · 실행 중 갱신 없음)
 1) 의미 갈래: CLIP ViT-L/14@336 시각부(구조 vit.py, 가중치 model/b2_visual_fp16.safetensors)로 원해상도 336×336 조각 3×3(9개) 임베딩
    → 조각별 L2 정규화 → 평균 → 표준화 + 로지스틱 회귀(model/b2_lr.npz) → p1
 2) 저수준 갈래: 잡음 잔차·선명도·JPEG 격자·채도·주파수 등 38개 통계(lowfeat.py) → 결정나무 부스팅(model/low_hgb.npz, numpy로 계산) → p2
 3) p = (p1 + p2) / 2
 4) 원본 대조 규칙(L2): 이미지 12×16 색 지문과 AI허브 공개 데이터 지문 DB(model/fp_db.npz)의 최소 평균 절대차 d1·2위 d2,
    gap = d2 − d1 에 대해 3.5 < d1 < 20 이고 gap > 8 이면 p ← max(p, 0.9)
 예외 대체(고정): 저수준 실패 → p1, CLIP 실패 → p2, 둘 다 실패 → 0.5, NaN → 0.5.
 판정 기준은 운영위원회 0.5(5)의 변환 후 확률 기준).
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


def _grid_crops(img):
    w, h = img.size
    out = []
    for yy in np.linspace(0, h - S, 3).astype(int):
        for xx in np.linspace(0, w - S, 3).astype(int):
            out.append(img.crop((int(xx), int(yy), int(xx) + S, int(yy) + S)))
    return out


def _fingerprint(img):
    """배포 형식(768x1024 또는 1024x768) 비율로 덮게 자른 뒤 BOX 평균 12x16(세로 기준) RGB."""
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


def load_model(model_dir=None):
    from safetensors.torch import load_file
    model_dir = Path(model_dir) if model_dir else HERE / "model"
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual = ViTL14_336()   # 구조만 생성(vit.py, 다운로드 없음)
    visual.load_state_dict({k: v.float() for k, v in load_file(str(model_dir / "b2_visual_fp16.safetensors")).items()})
    visual = visual.eval().to(dev)
    if dev.type == "cuda":
        visual = visual.half()
    lr = dict(np.load(model_dir / "b2_lr.npz"))
    hg = dict(np.load(model_dir / "low_hgb.npz"))
    db = np.load(model_dir / "fp_db.npz")
    fp, port = db["fp"].reshape(len(db["fp"]), -1), db["portrait"].astype(bool)
    dbt = {p: torch.from_numpy(fp[port == p]).to(dev) for p in (True, False)}
    return dict(visual=visual, dev=dev, lr=lr, hg=hg, db=dbt, cfg=cfg)


@torch.no_grad()
def _embed(m, imgs):
    xs = [(torch.from_numpy(np.asarray(c, np.float32) / 255.0).permute(2, 0, 1) - MEAN) / STD for im in imgs for c in _grid_crops(im)]
    x = torch.stack(xs).to(m["dev"])
    if m["dev"].type == "cuda":
        x = x.half()
    e = m["visual"](x).float().cpu().numpy().reshape(len(imgs), 9, -1)
    e = e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-8)
    return e.mean(1)


@torch.no_grad()
def _lookup(m, fpv, port):
    D = m["db"][port]
    q = torch.from_numpy(fpv).to(m["dev"])
    d = torch.empty(D.shape[0], device=m["dev"])
    for i in range(0, D.shape[0], 32768):
        d[i:i + 32768] = (D[i:i + 32768].float() - q).abs().mean(1)
    two = torch.topk(d, 2, largest=False).values.cpu().numpy()
    return float(two[0]), float(two[1])


@torch.no_grad()
def _one(m, path, low):
    """이미지 1장 확률. 대체 규칙(고정): 저수준 실패 → p1, CLIP 실패 → p2, 둘 다 실패 → 0.5. 원본 대조 실패 → 규칙 미적용."""
    lr, rule = m["lr"], m["cfg"]["l2_rule"]
    p1 = p2 = None
    img = None
    try:
        img = Image.open(path).convert("RGB")
        emb = _embed(m, [img])        # 항상 이미지 1장(9조각)씩 계산 → 같이 들어온 이미지·순서와 무관
        z = float(((emb[0] - lr["mean"]) / lr["scale"]) @ lr["coef"] + lr["intercept"][0])
        p1 = 1.0 / (1.0 + np.exp(-z))
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
            if rule["d1_lo"] < d1 < rule["d1_hi"] and (d2 - d1) > rule["gap"]:
                p = max(p, rule["floor"])
    except Exception:
        pass
    p = float(np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0))
    p = min(max(p, 0.0), 1.0)
    tau = float(m["cfg"].get("tau", 0.5))
    p = 0.5 * p / tau if p < tau else 0.5 + 0.5 * (p - tau) / (1.0 - tau)
    return min(max(p, 0.0), 1.0)


def _safe_low(path):
    try:
        v = low_features(path)
        return v if np.all(np.isfinite(v)) else None
    except Exception:
        return None


def predict(model, image_paths):
    workers = max(1, min(8, os.cpu_count() or 1))
    with ThreadPoolExecutor(workers) as ex:      # 저수준 특징은 CPU 스레드 병렬(이미지별 독립 계산)
        lows = list(ex.map(_safe_low, image_paths))
    return [_one(model, p, lw) for p, lw in zip(image_paths, lows)]


if __name__ == "__main__":
    paths = sorted(str(p) for p in Path(sys.argv[1]).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    mm = load_model()
    for p, pr in zip(paths, predict(mm, paths)):
        print(f"{Path(p).name}  {pr:.4f}")
