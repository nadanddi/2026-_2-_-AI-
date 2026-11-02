"""IM05b: v2 자료(프로그램 가짜 im_set_v2 + SD 가짜 im_set_sd2) 판별기 평가. 실행 전 고정한 검증:
 V1  : farm_code GroupKFold(5) — 같은 원본·같은 농장은 한 폴드(SD ai_gen은 GEN0~4).
 V1b : leave-one-family-out — 가짜 계열 하나를 학습에서 빼고, 그 계열 가짜 + 해당 폴드 실제로 평가(처음 보는 제작 방식 일반화).
 V2  : 공개 21장(학습 미사용) — 전체 학습 모델의 예측. 큰 실패 감지용.
모델: 'clip'(CLIP 로지스틱), 'low'(저수준 HistGB), 'stack'(두 확률 평균). 클래스 가중 균형(실제:가짜=1:1). 판정 0.5 고정.
사용: python im05b_eval_v2.py [models=clip,low,stack] [clip_agg=P]
"""
import sys, csv, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

HERE = Path(__file__).resolve().parent; LOCAL = HERE.parent / "local"
PUB = HERE.parents[2] / "공용/대회자료/이미지데이터/참가자_배포/공개샘플"
FAMILY = {"filter_blur": "filter", "filter_noise": "filter", "filter_sp": "filter", "filter_jpeg": "filter", "composite": "composite",
          "full_style": "pstyle", "partial_style": "pstyle", "partial_gen_sd": "sd_pgen", "partial_style_sd": "sd_pstyle",
          "full_style_sd": "sd_fstyle", "ai_gen": "sd_gen", "real": "real", "real_resampled": "real"}

def load_feats(name, kind):
    p = LOCAL / "feats" / f"{name}_{'clipL14' if kind == 'clip' else 'low'}.npz"
    if not p.exists(): return None
    z = np.load(p); F = z["feats"].astype(np.float32)
    if kind == "clip": F /= np.linalg.norm(F, axis=-1, keepdims=True)
    return dict(zip(list(z["files"]), F))

def clip_agg(F, mode):
    return {"G": F[:, 0], "P": F[:, 1:].mean(1), "GP": np.concatenate([F[:, 0], F[:, 1:].mean(1)], 1)}[mode]

RED_THR = 25  # 실제 원본의 평균(R-G) > 25 → 빨간 조명 원본(사용자 결정 10-10: 학습 제외, 진단만)

def red_sources():
    import numpy as _np
    from PIL import Image as _I
    cache = LOCAL / "red_sources.txt"
    if cache.exists(): return set(cache.read_text(encoding="utf-8").split(chr(10))) - {""}
    out = set()
    for r in csv.DictReader(open(LOCAL / "im_set_v2" / "manifest.csv", encoding="utf-8")):
        if r["kind"] != "real": continue
        a = _np.asarray(_I.open(LOCAL / "im_set_v2" / "img" / r["file"]).reduce(16), dtype=float)
        if (a[..., 0] - a[..., 1]).mean() > RED_THR: out.add(r["source"])
    cache.write_text(chr(10).join(sorted(out)), encoding="utf-8"); return out

def build(sets, use):
    rows, X = [], {u: [] for u in use}
    for s in sets:
        man = list(csv.DictReader(open(LOCAL / s / "manifest.csv", encoding="utf-8")))
        fs = {u: load_feats(s, u) for u in use}
        for r in man:
            if any(fs[u] is None or r["file"] not in fs[u] for u in use): continue
            rows.append(r)
            for u in use: X[u].append(fs[u][r["file"]])
    return rows, {u: np.stack(v) for u, v in X.items()}

def model(kind):
    if kind == "clip": return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=4000))
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)

def fitpred(X, y, tr, te_X, use, agg):
    out = {}
    w = compute_sample_weight("balanced", y[tr])
    for u in use:
        A = clip_agg(X[u], agg) if u == "clip" else X[u]
        B = clip_agg(te_X[u], agg) if u == "clip" else te_X[u]
        m = model(u)
        if u == "clip": m.fit(A[tr], y[tr], logisticregression__sample_weight=w)
        else: m.fit(A[tr], y[tr], sample_weight=w)
        out[u] = m.predict_proba(B)[:, 1]
    if len(use) > 1: out["stack"] = np.mean([out[u] for u in use], 0)
    return out

def rep(name, y, p, kind):
    s = f"{name:6s} acc={((p >= .5) == y).mean():.3f} auc={roc_auc_score(y, p):.3f} | "
    return s + " ".join(f"{k}={((p[kind == k] >= .5) == y[kind == k]).mean():.2f}" for k in sorted(set(kind)))

def main(use=("clip", "low"), agg="P"):
    sets = ["im_set_v2", "im_set_sd2"]
    rows_all, X_all = build(sets, use)
    red = red_sources()
    isred = np.array([r["source"] in red for r in rows_all])
    print(f"빨간 조명 원본 {len(red)}개 → 관련 행 {isred.sum()}개 학습·CV 제외(진단만)")
    rows = [r for r, k in zip(rows_all, ~isred) if k]; X = {u: v[~isred] for u, v in X_all.items()}
    Xred = {u: v[isred] for u, v in X_all.items()}; red_rows = [r for r, k in zip(rows_all, isred) if k]
    y = np.array([int(r["label"]) for r in rows]); kind = np.array([r["kind"] for r in rows]); fam = np.array([FAMILY[k] for k in kind])
    grp = np.array([r["farm_code"] for r in rows])
    print(f"n={len(y)} fake={y.sum()} kinds:", {k: int((kind == k).sum()) for k in sorted(set(kind))})
    oof = {}
    for tr, te in GroupKFold(5).split(y, y, grp):
        o = fitpred(X, y, tr, {u: X[u][te] for u in use}, use, agg)
        for k, v in o.items(): oof.setdefault(k, np.zeros(len(y)))[te] = v
    print("\n[V1 farm-group CV]"); [print(rep(k, y, v, kind)) for k, v in oof.items()]
    print("\n[V1b leave-one-family-out] (평가 = 그 계열 가짜 + 같은 폴드 실제)")
    for f in sorted(set(fam) - {"real"}):
        res = {}
        for tr, te in GroupKFold(5).split(y, y, grp):
            tr2 = tr[fam[tr] != f]; te2 = te[(fam[te] == f) | (fam[te] == "real")]
            if len(te2) == 0 or y[te2].sum() == 0: continue
            o = fitpred(X, y, tr2, {u: X[u][te2] for u in use}, use, agg)
            for k, v in o.items(): res.setdefault(k, []).append((y[te2], v))
        print(f"  {f:10s} " + " ".join(f"{k}: fake-recall={np.mean(np.concatenate([v for yy, v in r])[np.concatenate([yy for yy, v in r]) == 1] >= .5):.2f}"
                                        for k, r in res.items()))
    # V2 공개 21
    lab = {r["file"]: r for r in csv.DictReader(open(PUB / "sample_labels.csv"))}
    pf = list(lab); PX = {}
    for u in use:
        fz = load_feats("public21", u); PX[u] = np.stack([fz[f] for f in pf])
    py = np.array([int(lab[f]["label"]) for f in pf]); pk = np.array([lab[f]["type"] for f in pf])
    o = fitpred(X, y, np.arange(len(y)), PX, use, agg)
    print("\n[V2 public21]")
    for k, v in o.items():
        print(f"  {k:6s} acc={((v >= .5) == py).mean():.3f} auc={roc_auc_score(py, v):.3f} real_ok={(v[py == 0] < .5).sum()}/3  " +
              " ".join(f"{f[:4]}({t[:5]})={p:.2f}" for f, t, p in zip(pf, pk, v)))

if __name__ == "__main__":
    use = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("clip", "low")
    main(use, sys.argv[2] if len(sys.argv) > 2 else "P")
