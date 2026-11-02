"""IM20: 이미지 제출 01 최종 학습·내보내기 — 구성 ② B2 + 5판 + L2 (사용자 '제출해보자' 10-12, 검증 구성 그대로).
학습 자료: im_set_v4 + im_set_sd2 + im_set_v5(5판: filter_blur, composite_v5b, ai_gen_xl, full_style_xl, partial_style_xl, partial_gen_xl), 빨간 조명 원본 제외, tile_whole 제외.
모델: (1) B2 = CLIP ViT-L/14@336(openai) 시각부, 보기 = 원해상도 336 조각 3x3(각 임베딩 L2 정규화 후 평균) → 표준화 + 로지스틱(C=.05, 가중치 = 실제 .5 / 6유형 균등)
      (2) 저수준 38특징(im06) → HistGradientBoosting(max_iter 300, lr .05, leaf 15, l2 1, seed 0, 같은 가중치)
      (3) p = (p1 + p2)/2 → L2 규칙: 지문(12x16 RGB) d1∈(3.5,20) 이고 gap>8 이면 p ← max(p, .9)
출력(제출 model/): b2_visual_fp16.safetensors, b2_lr.npz, low_hgb.npz, fp_db.npz, config.json  +  검증용 공개 21장 예측
"""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05c_eval_v3 as V3  # noqa
V3.TYPE.update({"composite_v5b": "composite", "ai_gen_xl": "AI_gen", "full_style_xl": "full_style", "partial_style_xl": "partial_style", "partial_gen_xl": "partial_gen"})
import im05d_eval_v4 as D  # noqa
from im05k_backbones import bb  # noqa

LOCAL = D.LOCAL; OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else LOCAL / "submit_img01" / "model"

def export_hgb(clf):
    trees = []
    for it in clf._predictors:
        nd = it[0].nodes
        trees.append(np.stack([nd["feature_idx"].astype(np.float64), nd["num_threshold"], nd["left"].astype(np.float64), nd["right"].astype(np.float64),
                               nd["is_leaf"].astype(np.float64), nd["value"]], 1))
    lens = np.array([len(t) for t in trees]); flat = np.concatenate(trees)
    return dict(nodes=flat, lens=lens, base=np.array([float(np.ravel(clf._baseline_prediction)[0])]))

def hgb_predict(ex, X):
    raw = np.full(len(X), ex["base"][0]); off = 0
    for L in ex["lens"]:
        t = ex["nodes"][off:off + L]; off += L; node = np.zeros(len(X), int)
        for _ in range(64):
            leaf = t[node, 4] > 0
            if leaf.all(): break
            f = t[node, 0].astype(int); go_left = X[np.arange(len(X)), f] <= t[node, 1]
            nxt = np.where(go_left, t[node, 2], t[node, 3]).astype(int); node = np.where(leaf, node, nxt)
        raw += t[node, 5]
    return 1 / (1 + np.exp(-raw))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, X, isred = D.load(["im_set_v4", "im_set_sd2", "im_set_v5"]); keep = ~isred; R = [r for r, k in zip(rows, keep) if k]
    fz = {}
    for sn in ("im_set_v4", "im_set_sd2", "im_set_v5"): fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R]); Xl = X["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); w = V3.weights(kind); idx = np.arange(len(R))
    print(f"학습 행 {len(R)} (실제 {(y == 0).sum()}, 가짜 {y.sum()})", flush=True)
    lr = D.fit("clip", Xb, y, w, idx, 0); hg = D.fit("low", Xl, y, w, idx, 0)
    sc, lg = lr[0], lr[-1]
    np.savez(OUT / "b2_lr.npz", mean=sc.mean_, scale=sc.scale_, coef=lg.coef_[0], intercept=lg.intercept_)
    ex = export_hgb(hg); np.savez(OUT / "low_hgb.npz", **ex)
    # 내보낸 숫자 모델 = sklearn 결과 확인
    d_lr = np.abs(1 / (1 + np.exp(-(((Xb - sc.mean_) / sc.scale_) @ lg.coef_[0] + lg.intercept_[0]))) - lr.predict_proba(Xb)[:, 1]).max()
    d_hg = np.abs(hgb_predict(ex, Xl[:2000]) - hg.predict_proba(Xl[:2000])[:, 1]).max()
    print(f"내보내기 검산: LR 최대차 {d_lr:.2e}, HGB 최대차 {d_hg:.2e}", flush=True); assert d_lr < 1e-6 and d_hg < 1e-6
    # B2 시각부 가중치 fp16
    import open_clip
    from safetensors.torch import save_file
    m, _, _ = open_clip.create_model_and_transforms("ViT-L-14-336", pretrained="openai", cache_dir=str(LOCAL / "weights/openclip"))
    save_file({k: v.half().contiguous() for k, v in m.visual.state_dict().items()}, str(OUT / "b2_visual_fp16.safetensors"))
    # 지문 DB(세로 기준 12x16x3, 방향 플래그)
    names, fps, ports = [], [], []
    for p in sorted((LOCAL / "fp_db").glob("*.npz")):
        z = np.load(p); fps.append(z["fp"]); ports.append(z["portrait"])
    np.savez_compressed(OUT / "fp_db.npz", fp=np.concatenate(fps).astype(np.uint8), portrait=np.concatenate(ports))
    cfg = dict(version="img01", backbone="open_clip ViT-L-14-336 (openai), visual tower fp16", view=336, grid=3, combine="mean(p_b2, p_low)",
               l2_rule=dict(d1_lo=3.5, d1_hi=20.0, gap=8.0, floor=0.9), threshold=0.5, train_rows=int(len(R)))
    (OUT / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    # 검증용: 공개 21장 예측(학습 특징 경로) 저장 → 패키지 추론과 대조
    lab, pf, PX = D.pub_feats(); pz = bb("public21", "B2"); Pb = np.stack([pz[f] for f in pf])
    p = (lr.predict_proba(Pb)[:, 1] + hg.predict_proba(PX["low"])[:, 1]) / 2
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pf])
    hit = (PL[:, 0] > 3.5) & (PL[:, 0] < 20) & (PL[:, 2] > 8); pr = np.where(hit, np.maximum(p, .9), p)
    np.savez(OUT.parent / "offline_public21.npz", files=np.array(pf), p_stack=p, p_final=pr)
    print("공개 21장(오프라인 경로): " + " ".join(f"{f[:4]}={q:.3f}" for f, q in zip(pf, pr)))
    print("완료", sorted(x.name for x in OUT.iterdir()))

if __name__ == "__main__":
    main()
