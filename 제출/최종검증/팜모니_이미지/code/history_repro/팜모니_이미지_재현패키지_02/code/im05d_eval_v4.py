"""IM05d: 평가 v4 — im05c(v3)의 중간 비평 지적 수정판. 규칙(계획 v3 §2·§3)은 같고 구현만 고침.
수정:
 - 시드가 실제로 다르게: 시드 s마다 (a) 폴드 배정 순열, (b) 공개 21장 예측용 전체 모델을 farm_code 90% 부분표본으로 학습(s별 다른 표본), (c) HistGB random_state.
 - CNN 예측이 없는 행 때문에 기준 학습 행이 줄지 않게: 학습은 항상 전체 행. 보조 평가 행만 '비교하는 모든 모델의 OOF가 있는 행'으로 제한.
 - 부트스트랩 분모 = 유효 표본 수.
 - avg: 결합은 성분마다 OOF로 학습한 Platt 보정(로짓→확률) 후 평균(보정 상태 차이 제거).
 - 보조 V3는 '비악화 확인'용(비평 중요-3). 판정 규칙은 그대로 적용하되 보고에 명시.
특징 갈래: clip(패치 평균), low(38), reg(영역 불일치 27, im06b), het(CLIP 패치 이질성 6).
모델: clip, low, lowA(=low+reg+het, HistGB), stack(=clip·low 평균, 기준), stackA(=clip·lowA 평균, 후보 A), cnn_*, avg:a+b
사용: python im05d_eval_v4.py <models> [base] [sets=im_set_v4,im_set_sd2]
"""
import sys, csv, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05c_eval_v3 as V3  # noqa  (TYPE, TYPES6, weights, folds, bal_acc 재사용)

LOCAL, PUB = V3.LOCAL, V3.PUB
TYPE, TYPES6 = V3.TYPE, V3.TYPES6
N_TYPE, N_REAL, SEEDS, NB = 40, 240, (0, 1, 2), 2000

def raw(name, kind):
    p = LOCAL / "feats" / f"{name}_{ {'clip': 'clipL14', 'low': 'low', 'reg': 'reg'}[kind] }.npz"
    if not p.exists(): return None
    z = np.load(p); return dict(zip(list(z["files"]), z["feats"].astype(np.float32)))

def clip_parts(F):
    F = F / np.linalg.norm(F, axis=-1, keepdims=True); P = F[:, 1:]; m = P.mean(1); mn = m / np.linalg.norm(m, axis=1, keepdims=True)
    s = np.einsum("npd,nd->np", P, mn); pair = np.einsum("npd,nqd->npq", P, P); iu = np.triu_indices(P.shape[1], 1)
    pw = pair[:, iu[0], iu[1]]
    het = np.stack([s.min(1), s.mean(1), s.std(1), pw.min(1), pw.mean(1), np.einsum("nd,nd->n", F[:, 0], mn)], 1)
    return m, het

def load(sets):
    red = set((LOCAL / "red_sources.txt").read_text(encoding="utf-8").split(chr(10))) - {""}
    rows, C, L, G, H, isred = [], [], [], [], [], []
    for s in sets + ["im_set_v2"]:
        fc, fl, fg = raw(s, "clip"), raw(s, "low"), raw(s, "reg")
        for r in csv.DictReader(open(LOCAL / s / "manifest.csv", encoding="utf-8")):
            if "tile_whole=1" in r["info"]: continue
            if s == "im_set_v2":
                if r["source"] not in red: continue
                r = dict(r, kind={"full_style": "pstyle_color"}.get(r["kind"], r["kind"]))
            if r["kind"] not in TYPE: continue
            if any(d is None or r["file"] not in d for d in (fc, fl, fg)): continue
            rows.append(dict(r, _set=s)); isred.append(r["source"] in red)
            C.append(fc[r["file"]]); L.append(fl[r["file"]]); G.append(fg[r["file"]])
    C = np.stack(C); m, het = clip_parts(C)
    return rows, {"clip": m, "low": np.stack(L), "lowA": np.concatenate([np.stack(L), np.stack(G), het], 1)}, np.array(isred)

def pub_feats():
    lab = list(csv.DictReader(open(PUB / "sample_labels.csv"))); pf = [r["file"] for r in lab]
    fc, fl, fg = raw("public21", "clip"), raw("public21", "low"), raw("public21", "reg")
    m, het = clip_parts(np.stack([fc[f] for f in pf]))
    L = np.stack([fl[f] for f in pf]); G = np.stack([fg[f] for f in pf])
    return lab, pf, {"clip": m, "low": L, "lowA": np.concatenate([L, G, het], 1)}

def mdl(u, seed):
    if u == "clip": return make_pipeline(StandardScaler(), LogisticRegression(C=.05, max_iter=4000))
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1., random_state=seed)

def fit(u, X, y, w, idx, seed):
    m = mdl(u, seed); kw = {"logisticregression__sample_weight": w[idx]} if u == "clip" else {"sample_weight": w[idx]}
    return m.fit(X[idx], y[idx], **kw)

def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

def main(models, base=None, sets=("im_set_v4", "im_set_sd2")):
    parts = set(sum([m[4:].split("+") if m.startswith("avg:") else [m] for m in models], []))
    need = {"clip"} | {"low"} | {"lowA"}
    rows, X, isred = load(list(sets))
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Rr = [r for r, k in zip(rows, isred) if k]
    Xk = {u: v[keep] for u, v in X.items()}; Xr = {u: v[isred] for u, v in X.items()}
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); w = V3.weights(kind)
    lab, pf, PX = pub_feats(); py = np.array([int(r["label"]) for r in lab]); pt = np.array([r["type"] for r in lab])
    preds = {}
    for m in parts:
        if m.startswith("cnn_"):
            z = np.load(LOCAL / "preds" / f"{m}.npz"); o = dict(zip(z["files"], z["oof"]))
            preds[m] = (np.array([o.get(f"{r['_set']}/{r['file']}", np.nan) for r in R]), dict(zip(z["pub_files"], z["pub"])))
    avail = np.ones(len(R), bool)
    for m, (o, _) in preds.items(): avail &= ~np.isnan(o)
    print(f"학습 행 {len(R)} (빨간 조명 진단 {len(Rr)}행) | 보조 평가 가능 행 {avail.sum()} | " + " ".join(f"{x}={(t == x).sum()}" for x in ["real"] + TYPES6))
    # 보조 V3 선택: CLIP 패치평균 최근접, 평가 가능 행 안에서
    S = Xk["clip"] / np.linalg.norm(Xk["clip"], axis=1, keepdims=True); PS = PX["clip"] / np.linalg.norm(PX["clip"], axis=1, keepdims=True)
    aux = []
    for x, n in [("real", N_REAL)] + [(x, N_TYPE) for x in TYPES6]:
        cand = np.where((t == x) & avail)[0]
        if len(cand): aux += list(cand[np.argsort(-(S[cand] @ PS[pt == x].T).max(1))[:n]])
    aux = np.array(aux)
    res = {}
    for s in SEEDS:
        oof, full, red = {}, {}, {}
        rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp)))
        bag = np.where(np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False)))[0]
        for u in ("clip", "low", "lowA"):
            if u not in parts and not ({"stack", "stackA"} & parts): continue
            o = np.zeros(len(y))
            for tr, te in V3.folds(grp, s): o[te] = fit(u, Xk[u], y, w, tr, s).predict_proba(Xk[u][te])[:, 1]
            oof[u] = o; mf = fit(u, Xk[u], y, w, bag, s); full[u] = mf.predict_proba(PX[u])[:, 1]
            if len(Rr): red[u] = mf.predict_proba(Xr[u])[:, 1]
        for a, b, nm in [("clip", "low", "stack"), ("clip", "lowA", "stackA")]:
            if a in oof and b in oof:
                oof[nm] = (oof[a] + oof[b]) / 2; full[nm] = (full[a] + full[b]) / 2
                if red: red[nm] = (red[a] + red[b]) / 2
        for m, (o, pp) in preds.items(): oof[m] = o; full[m] = np.array([pp[f] for f in pf])
        for m in models:
            if m.startswith("avg:"):
                cs = m[4:].split("+"); ok = avail & np.all([~np.isnan(oof[c]) for c in cs], 0); lo, lf = [], []
                for c in cs:
                    cal = LogisticRegression().fit(lg(oof[c][ok])[:, None], y[ok])
                    lo.append(np.where(np.isnan(oof[c]), np.nan, cal.predict_proba(lg(np.nan_to_num(oof[c], nan=.5))[:, None])[:, 1]))
                    lf.append(cal.predict_proba(lg(full[c])[:, None])[:, 1])
                oof[m] = np.mean(lo, 0); full[m] = np.mean(lf, 0)
        res[s] = (oof, full, red)
    yr = np.array([int(r["label"]) for r in Rr]) if Rr else None
    print("\n== 모델별 (시드 0/1/2, 공개 예측은 시드별 90% 농장 부분표본 모델) ==")
    for m in models:
        print(f"[{m}]")
        for s in SEEDS:
            oof, full, red = res[s]; p, o = full[m], oof[m]
            hard = np.isin(pt, ["composite", "full_style", "partial_gen", "partial_style"])
            per = " ".join(f"{x[:6]}={((p[pt == x] >= .5) == (x != 'real')).sum()}" for x in ["real"] + TYPES6)
            print(f"  s{s} 공개: {((p >= .5) == py).sum()}/21 logloss={log_loss(py, np.clip(p, 1e-4, 1 - 1e-4)):.3f} auc={roc_auc_score(py, p):.3f} "
                  f"실제오탐={(p[py == 0] >= .5).sum()} 어려운12={(p[hard] >= .5).sum()} | {per}")
            print(f"      보조(비악화 확인) 균형acc={V3.bal_acc(o[aux], y[aux], t[aux]):.3f} | " +
                  " ".join(f"{x[:6]}={((o[aux][t[aux] == x] >= .5) == (x != 'real')).mean():.2f}" for x in ["real"] + TYPES6) +
                  (f" | 빨강 실제정답={(red[m][yr == 0] < .5).mean():.2f}" if m in red else ""))
    if base:
        k = max(1, len(models) - 1); thr = .025 / k
        print(f"\n== 채택 규칙 (기준={base}, 후보 {k}개 → 문턱 {thr:.4f}) ==")
        fa = grp[aux]; farms = np.array(sorted(set(fa)))
        for m in models:
            if m == base: continue
            ok2, dirs, pw = [], [], []
            for s in SEEDS:
                oof, full, _ = res[s]; pb, pc = full[base], full[m]
                ll = lambda q: log_loss(py, np.clip(q, 1e-4, 1 - 1e-4))
                ok2.append(bool(((pc >= .5) == py).sum() >= ((pb >= .5) == py).sum() and ll(pc) < ll(pb) and (pc[py == 0] >= .5).sum() <= (pb[py == 0] >= .5).sum()))
                ob, oc = oof[base][aux], oof[m][aux]; ya, ta = y[aux], t[aux]
                dirs.append(bool(V3.bal_acc(oc, ya, ta) > V3.bal_acc(ob, ya, ta)))
                rng = np.random.default_rng(100 + s); worse = valid = 0
                for _ in range(NB):
                    pick = np.concatenate([np.where(fa == f)[0] for f in rng.choice(farms, len(farms))])
                    if (ya[pick] == 0).sum() == 0 or (ya[pick] == 1).sum() == 0: continue
                    valid += 1; worse += V3.bal_acc(oc[pick], ya[pick], ta[pick]) <= V3.bal_acc(ob[pick], ya[pick], ta[pick])
                pw.append(worse / max(valid, 1))
            v = all(ok2) and all(dirs) and max(pw) < thr
            print(f"  {m}: 공개조건 {ok2} | 보조방향 {dirs} | p_worse {[round(x, 4) for x in pw]} → {'채택' if v else '불채택'}")

if __name__ == "__main__":
    ms = sys.argv[1].split(","); b = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    main(ms, b, tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("im_set_v4", "im_set_sd2"))
