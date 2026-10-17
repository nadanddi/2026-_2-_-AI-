"""IM05c: 평가 v3 (계획 v3 §2·§3 고정 규칙 구현).
 중심 V2 = 공개 21장(학습 미사용): 정확도·로그손실·AUC·실제 오탐·유형별
 보조 V3 = 공개 샘플을 닮은 우리 자료(유형별 40 + 실제 240, CLIP 패치평균 코사인 최근접), farm 그룹 OOF, 균형 정확도
 시드 0/1/2 = 폴드 배정 순열 + HistGB random_state.  farm 군집 부트스트랩 2000회.
 학습 가중: 실제 0.5 / 가짜 0.5, 가짜 안에서는 공개 6유형 균등, 실제 안에서 리샘플 실제 합 = 확대 가짜 비율.
 진단: 빨간 조명 행, 전체 OOF 균형 정확도.
사용: python im05c_eval_v3.py <models 쉼표> [base] [feat_for_sim=clip|low]
  models 예: clip,low,stack  (stack = clip·low 확률 평균).  base가 주어지면 나머지 각각을 base와 규칙 비교.
"""
import sys, csv, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).resolve().parent; LOCAL = HERE.parent / "local"
PUB = HERE.parents[2] / "공용/대회자료/이미지데이터/참가자_배포/공개샘플"
SETS = ["im_set_v4", "im_set_sd2"]   # 10-11: 합성 v4
TYPE = {"ai_gen": "AI_gen", "composite": "composite", "filter_blur": "filter", "filter_noise": "filter", "filter_sp": "filter",
        "filter_jpeg": "filter", "pstyle_color": "full_style", "full_style_sd": "full_style", "partial_gen_sd": "partial_gen",
        "partial_style": "partial_style", "partial_style_sd": "partial_style", "real": "real", "real_resampled": "real"}
UPSCALED = {"ai_gen", "full_style_sd"}
TYPES6 = ["AI_gen", "composite", "filter", "full_style", "partial_gen", "partial_style"]
N_TYPE, N_REAL, SEEDS, NB = 40, 240, (0, 1, 2), 2000

def feats(name, kind):
    p = LOCAL / "feats" / f"{name}_{'clipL14' if kind == 'clip' else 'low'}.npz"
    if not p.exists(): return None
    z = np.load(p); F = z["feats"].astype(np.float32)
    if kind == "clip": F /= np.linalg.norm(F, axis=-1, keepdims=True); F = F[:, 1:].mean(1)
    return dict(zip(list(z["files"]), F))

def load(use):
    red = set((LOCAL / "red_sources.txt").read_text(encoding="utf-8").split(chr(10))) - {""}
    rows, X, isred = [], {u: [] for u in use}, []
    for s in SETS + ["im_set_v2"]:   # im_set_v2는 빨간 조명 진단 행만 가져옴
        fs = {u: feats(s, u) for u in use}
        for r in csv.DictReader(open(LOCAL / s / "manifest.csv", encoding="utf-8")):
            if "tile_whole=1" in r["info"]: continue
            if s == "im_set_v2":
                if r["source"] not in red: continue
                r = dict(r, kind={"full_style": "pstyle_color"}.get(r["kind"], r["kind"]))
            if r["kind"] not in TYPE: continue
            if any(fs[u] is None or r["file"] not in fs[u] for u in use): continue
            rows.append(dict(r, _set=s)); isred.append(r["source"] in red)
            for u in use: X[u].append(fs[u][r["file"]])
    return rows, {u: np.stack(v) for u, v in X.items()}, np.array(isred)

def weights(kind):
    t = np.array([TYPE[k] for k in kind]); w = np.zeros(len(kind))
    fake = t != "real"; ups = np.isin(kind, list(UPSCALED)); rs = kind == "real_resampled"
    present = [x for x in TYPES6 if (t == x).any()]
    for x in present: w[t == x] = .5 / len(present) / (t == x).sum()
    up_share = sum(w[ups]) / .5 if fake.any() else 0
    if rs.any(): w[rs] = .5 * up_share / rs.sum()
    plain = (t == "real") & ~rs
    w[plain] = .5 * (1 - up_share) / plain.sum()
    return w * len(kind)

def mdl(u, seed):
    if u == "clip": return make_pipeline(StandardScaler(), LogisticRegression(C=.05, max_iter=4000))
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1., random_state=seed)

def fitpred(X, y, w, tr, TE, use, seed):
    out = {}
    for u in use:
        m = mdl(u, seed); kw = {"logisticregression__sample_weight": w[tr]} if u == "clip" else {"sample_weight": w[tr]}
        m.fit(X[u][tr], y[tr], **kw); out[u] = m.predict_proba(TE[u])[:, 1]
    if "clip" in use and "low" in use: out["stack"] = (out["clip"] + out["low"]) / 2
    return out

def folds(grp, seed, k=5):
    g = np.array(sorted(set(grp))); rng = np.random.default_rng(seed); rng.shuffle(g)
    a = {x: i % k for i, x in enumerate(g)}; f = np.array([a[x] for x in grp])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]

def bal_acc(p, y, t):
    r = (p[y == 0] < .5).mean(); rec = [(p[t == x] >= .5).mean() for x in TYPES6 if (t == x).any()]
    return .5 * r + .5 * np.mean(rec)

def load_pred(name):
    z = np.load(LOCAL / "preds" / f"{name}.npz")
    return (dict(zip(z["files"], z["oof"])), dict(zip(z["pub_files"], z["pub"])), dict(zip(z["red_files"], z["red"])))

def main(models, base=None, simfeat="clip"):
    # 모델 이름: clip, low, stack(=clip·low 평균), cnn_*(미리 계산한 예측 preds/<이름>.npz), avg:a+b+c(확률 평균)
    parts = set(sum([m[4:].split("+") if m.startswith("avg:") else [m] for m in models], []))
    use = [u for u in ("clip", "low") if u in parts or "stack" in parts]
    preds = {m: load_pred(m) for m in parts if m.startswith("cnn_")}
    rows, X, isred = load(use)
    keep = ~isred
    if preds:   # 예측이 있는 행만(학습 시점 이후 추가된 SD 행 제외)
        has = np.array([all(f"{r['_set']}/{r['file']}" in P[0] and not np.isnan(P[0][f"{r['_set']}/{r['file']}"]) for P in preds.values()) or rr
                        for r, rr in zip(rows, isred)])
        keep = keep & has
    R = [r for r, k in zip(rows, keep) if k]
    Xk = {u: v[keep] for u, v in X.items()}; Xr = {u: v[isred] for u, v in X.items()}; Rr = [r for r, k in zip(rows, isred) if k]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); w = weights(kind)
    print(f"학습 행 {len(R)} (빨간 조명 {isred.sum()}행 제외) | " + " ".join(f"{x}={(t == x).sum()}" for x in ["real"] + TYPES6))
    # 공개 21장
    lab = {r["file"]: r for r in csv.DictReader(open(PUB / "sample_labels.csv"))}; pf = list(lab)
    PX = {u: np.stack([feats("public21", u)[f] for f in pf]) for u in use}
    py = np.array([int(lab[f]["label"]) for f in pf]); pt = np.array([lab[f]["type"].replace("real", "real") for f in pf])
    # 보조 V3 선택(공개 특징 최근접)
    S = Xk[simfeat] if simfeat in Xk else Xk[use[0]]; PS = PX[simfeat] if simfeat in PX else PX[use[0]]
    if simfeat == "low":
        mu, sd = S.mean(0), S.std(0) + 1e-6; S, PS = (S - mu) / sd, (PS - mu) / sd
    Sn = S / np.linalg.norm(S, axis=1, keepdims=True); PSn = PS / np.linalg.norm(PS, axis=1, keepdims=True)
    aux = []
    for x, n in [("real", N_REAL)] + [(x, N_TYPE) for x in TYPES6]:
        cand = np.where(t == x)[0]
        if not len(cand): continue
        sim = (Sn[cand] @ PSn[pt == x].T).max(1); aux += list(cand[np.argsort(-sim)[:n]])
    aux = np.array(aux)
    print(f"보조 V3: {len(aux)}행 = " + " ".join(f"{x}={(t[aux] == x).sum()}" for x in ["real"] + TYPES6))
    res = {}
    for s in SEEDS:
        oof = {}
        for tr, te in folds(grp, s):
            o = fitpred(Xk, y, w, tr, {u: Xk[u][te] for u in use}, use, s)
            for k, v in o.items(): oof.setdefault(k, np.zeros(len(y)))[te] = v
        full = fitpred(Xk, y, w, np.arange(len(y)), PX, use, s)
        red = fitpred(Xk, y, w, np.arange(len(y)), Xr, use, s) if len(Rr) else {}
        for m, (po, pp, pr) in preds.items():
            oof[m] = np.array([po[f"{r['_set']}/{r['file']}"] for r in R]); full[m] = np.array([pp[f] for f in pf])
            if red: red[m] = np.array([pr.get(f"{r['_set']}/{r['file']}", np.nan) for r in Rr])
        for m in models:
            if m.startswith("avg:"):
                ps = m[4:].split("+")
                for d in (oof, full) + ((red,) if red else ()): d[m] = np.mean([d[q] for q in ps], 0)
        res[s] = (oof, full, red)
    yr = np.array([int(r["label"]) for r in Rr]) if Rr else None
    print("\n== 모델별 (시드 0/1/2) ==")
    for m in models:
        print(f"[{m}]")
        for s in SEEDS:
            oof, full, red = res[s]; p = full[m]; o = oof[m]
            per = " ".join(f"{x[:6]}={((p[pt == x] >= .5) == (x != 'real')).sum()}" for x in ["real"] + TYPES6)
            print(f"  s{s} V2공개: acc={((p >= .5) == py).mean():.3f} ({((p >= .5) == py).sum()}/21) logloss={log_loss(py, np.clip(p, 1e-4, 1 - 1e-4)):.3f} "
                  f"auc={roc_auc_score(py, p):.3f} 실제오탐={(p[py == 0] >= .5).sum()} | {per}")
            print(f"      V3보조 균형acc={bal_acc(o[aux], y[aux], t[aux]):.3f} | " + " ".join(f"{x[:6]}={((o[aux][t[aux] == x] >= .5) == (x != 'real')).mean():.2f}" for x in ["real"] + TYPES6)
                  + f" || 전체OOF 균형acc={bal_acc(o, y, t):.3f}" + (f" | 빨강 실제정답={(red[m][yr == 0] < .5).mean():.2f} 빨강가짜탐지={(red[m][yr == 1] >= .5).mean():.2f}" if red else ""))
    if base:
        print(f"\n== 채택 규칙 비교 (기준={base}, 후보 {len(models) - 1}개 → 문턱 {.025 / max(1, len(models) - 1):.4f}) ==")
        fa = np.array([r["farm_code"] for r in R])[aux]; farms = np.array(sorted(set(fa)))
        for m in models:
            if m == base: continue
            ok2, dirs, pw = [], [], []
            for s in SEEDS:
                oof, full, _ = res[s]; pb, pc = full[base], full[m]
                ok2.append(((pc >= .5) == py).sum() >= ((pb >= .5) == py).sum()
                           and log_loss(py, np.clip(pc, 1e-4, 1 - 1e-4)) < log_loss(py, np.clip(pb, 1e-4, 1 - 1e-4))
                           and (pc[py == 0] >= .5).sum() <= (pb[py == 0] >= .5).sum())
                ob, oc = oof[base][aux], oof[m][aux]; ya, ta = y[aux], t[aux]
                dirs.append(bal_acc(oc, ya, ta) > bal_acc(ob, ya, ta))
                rng = np.random.default_rng(100 + s); worse = 0
                for _ in range(NB):
                    pick = np.concatenate([np.where(fa == f)[0] for f in rng.choice(farms, len(farms))])
                    if len(set(ta[pick])) < 2 or (ya[pick] == 0).sum() == 0: continue
                    worse += bal_acc(oc[pick], ya[pick], ta[pick]) <= bal_acc(ob[pick], ya[pick], ta[pick])
                pw.append(worse / NB)
            thr = .025 / max(1, len(models) - 1)
            verdict = all(ok2) and all(dirs) and max(pw) < thr
            print(f"  {m}: V2조건 {ok2} | V3 방향 {dirs} | p_worse {[round(x, 4) for x in pw]} → {'채택' if verdict else '불채택'}")

if __name__ == "__main__":
    ms = sys.argv[1].split(","); main(ms, sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None, sys.argv[3] if len(sys.argv) > 3 else "clip")
