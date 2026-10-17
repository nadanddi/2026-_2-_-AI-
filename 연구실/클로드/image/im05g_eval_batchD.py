"""IM05g: 후보 묶음 D 일괄 판정 (IM_후보묶음D_v1.md에 고정한 정의 그대로). 본페로니 k=5 → 문턱 .005.
기준 stack = (CLIP 패치평균 LR + 저수준 HistGB)/2.  자료 v4 + SD2, 빨간 조명 행은 진단만.
"""
import sys, csv, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, log_loss
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05d_eval_v4 as D, im05c_eval_v3 as V3  # noqa

LOCAL = D.LOCAL; SEEDS, NB, K = (0, 1, 2), 2000, 6; THR = .025 / K   # 계획 비평: D0 대조군 추가 → k=6
FAM = {"filter": ["filter"], "composite": ["composite"], "style": ["full_style", "partial_style"], "gen": ["AI_gen", "partial_gen"]}
TAU_FP = .015; D0_FP = .05

def l2(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
def lg(p): p = np.clip(p, 1e-5, 1 - 1e-5); return np.log(p / (1 - p))
def sg(z): return 1 / (1 + np.exp(-z))

def newfeats(rows):
    cache = {}
    def get(setname, kind):
        k = (setname, kind)
        if k not in cache:
            z = np.load(LOCAL / "feats" / f"{setname}_{kind}.npz"); z = {kk: z[kk] for kk in z.files}   # 한 번만 풀어 둠(행마다 재해제 방지)
            cache[k] = (dict(zip(z["files"], range(len(z["files"])))), z)
        return cache[k]
    D1, D2, D3 = [], [], []
    for r in rows:
        sn = "im_set_v2red" if r["_set"] == "im_set_v2" else r["_set"]
        ix, z = get(sn, "clipML"); i = ix[r["file"]]
        D1.append(np.concatenate([z["ml_patch"][i].astype(np.float32).ravel(), z["ml_glob"][i].astype(np.float32).ravel()]))   # ln_post 전 값: l2 없이 표준화만(비평 경미-7)
        D3.append(l2(z["patches"][i].astype(np.float32)))
        ix, z = get(sn, "dino"); f = z["feats"][ix[r["file"]]].astype(np.float32)
        D2.append(np.concatenate([l2(f[0]), l2(f[1:].mean(0))]))
    return np.stack(D1), np.stack(D2), np.stack(D3)

def pubnew(pf):
    z = np.load(LOCAL / "feats" / "public21_clipML.npz"); z = {kk: z[kk] for kk in z.files}; ix = dict(zip(z["files"], range(21)))
    d1 = np.stack([np.concatenate([z["ml_patch"][ix[f]].astype(np.float32).ravel(), z["ml_glob"][ix[f]].astype(np.float32).ravel()]) for f in pf])
    d3 = np.stack([l2(z["patches"][ix[f]].astype(np.float32)) for f in pf])
    z = np.load(LOCAL / "feats" / "public21_dino.npz"); z = {kk: z[kk] for kk in z.files}; ix = dict(zip(z["files"], range(21)))
    d2 = np.stack([np.concatenate([l2(z["feats"][ix[f]][0].astype(np.float32)), l2(z["feats"][ix[f]][1:].astype(np.float32).mean(0))]) for f in pf])
    return d1, d2, d3

def lr(C): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))

def fit_lr(X, y, w, idx, C): return lr(C).fit(X[idx], y[idx], logisticregression__sample_weight=w[idx])

def fit_patch(P, y, w, idx):
    n, k, d = P.shape; Xp = P[idx].reshape(-1, d); yp = np.repeat(y[idx], k); wp = np.repeat(w[idx], k) / k
    return lr(.05).fit(Xp, yp, logisticregression__sample_weight=wp)

def pred_patch(m, P, top=5):
    n, k, d = P.shape; p = m.predict_proba(P.reshape(-1, d))[:, 1].reshape(n, k); return np.sort(p, 1)[:, -top:].mean(1)

def expert_scores(Xc, Xl, y, w, t, tr, TE, s, grp):
    """D4: 계열 전문가 OR. tr 안에서 3겹 내부 OOF로 문턱 τ_F(실제 오탐 1.5%) → 테스트 점수 max_F sigmoid(logit p − logit τ)."""
    outs = []
    for F, types in FAM.items():
        sel = tr[(y[tr] == 0) | np.isin(t[tr], types)]
        inner = np.zeros(len(sel))
        for ia, ib in V3.folds(grp[sel], s, k=3):   # 계획 비평 치명-1: 농장 단위 내부 폴드
            a, b = sel[ia], sel[ib]
            inner[ib] = (D.fit("clip", Xc, y, w, a, s).predict_proba(Xc[b])[:, 1] + D.fit("low", Xl, y, w, a, s).predict_proba(Xl[b])[:, 1]) / 2
        tau = np.quantile(inner[y[sel] == 0], 1 - TAU_FP)
        p = (D.fit("clip", Xc, y, w, sel, s).predict_proba(TE[0])[:, 1] + D.fit("low", Xl, y, w, sel, s).predict_proba(TE[1])[:, 1]) / 2
        outs.append(lg(p) - lg(tau))
    return sg(np.max(outs, 0))

def main():
    rows, X, isred = D.load(["im_set_v4", "im_set_sd2"])
    D1, D2, D3 = newfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]
    F = {"clip": X["clip"], "low": X["low"], "d1": D1, "d2": D2, "d3": D3}
    Fk = {k: v[keep] for k, v in F.items()}; Fr = {k: v[isred] for k, v in F.items()}
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); w = V3.weights(kind); yr = np.array([int(r["label"]) for r, k in zip(rows, isred) if k])
    lab, pf, PX = D.pub_feats(); p1, p2, p3 = pubnew(pf)
    PF = {"clip": PX["clip"], "low": PX["low"], "d1": p1, "d2": p2, "d3": p3}
    py = np.array([int(r["label"]) for r in lab]); pt = np.array([r["type"] for r in lab])
    S, PS = l2(Fk["clip"]), l2(PF["clip"]); aux = []
    for x, n in [("real", 240)] + [(x, 40) for x in V3.TYPES6]:
        cand = np.where(t == x)[0]
        if len(cand): aux += list(cand[np.argsort(-(S[cand] @ PS[pt == x].T).max(1))[:n]])
    aux = np.array(aux)
    print(f"학습 행 {len(R)}, 빨간 진단 {len(yr)}, 보조 {len(aux)} | 공개 21장 열람 5번째(묶음 D 1회, 사용자 지시)", flush=True)

    def base_models(tr, TE, s):
        o = {"clip": D.fit("clip", Fk["clip"], y, w, tr, s).predict_proba(TE["clip"])[:, 1],
             "low": D.fit("low", Fk["low"], y, w, tr, s).predict_proba(TE["low"])[:, 1],
             "d1": fit_lr(Fk["d1"], y, w, tr, .01).predict_proba(TE["d1"])[:, 1],
             "d2": fit_lr(Fk["d2"], y, w, tr, .01).predict_proba(TE["d2"])[:, 1],
             "d3": pred_patch(fit_patch(Fk["d3"], y, w, tr), TE["d3"])}
        o["d4"] = expert_scores(Fk["clip"], Fk["low"], y, w, t, tr, (TE["clip"], TE["low"]), s, grp)
        # D0 대조군: stack 하나에 단일 문턱(학습 폴드 안 농장 그룹 내부 OOF에서 실제 오탐 5%) → sigmoid(logit p − logit τ)
        inner = np.zeros(len(tr))
        for ia, ib in V3.folds(grp[tr], s, k=3):
            inner[ib] = (D.fit("clip", Fk["clip"], y, w, tr[ia], s).predict_proba(Fk["clip"][tr[ib]])[:, 1] + D.fit("low", Fk["low"], y, w, tr[ia], s).predict_proba(Fk["low"][tr[ib]])[:, 1]) / 2
        tau = np.quantile(inner[y[tr] == 0], 1 - D0_FP)
        o["d0"] = sg(lg((o["clip"] + o["low"]) / 2) - lg(tau))
        return o

    res = {}
    for s in SEEDS:
        oof = {}
        for f, (tr, te) in enumerate(V3.folds(grp, s)):
            o = base_models(tr, {k: v[te] for k, v in Fk.items()}, s)
            for k, v in o.items(): oof.setdefault(k, np.zeros(len(y)))[te] = v
            print(f"  s{s} fold{f} 완료", flush=True)
        rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp)))
        bag = np.where(np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False)))[0]
        full = base_models(bag, PF, s); red = base_models(bag, Fr, s)
        # 결합
        for d in (oof, full, red):
            d["stack"] = (d["clip"] + d["low"]) / 2
            for c in ("d1", "d2"): d[f"{c}s"] = (d[c] + d["low"]) / 2
        cal = LogisticRegression().fit(lg(oof["d3"])[:, None], y)   # D3: OOF Platt 보정 후 결합(비평 중요-4)
        for d in (oof, full, red): d["d3s"] = (cal.predict_proba(lg(d["d3"])[:, None])[:, 1] + d["low"]) / 2
        # D5 학습형 결합기 (D4는 성분에 넣지 않음 — 기록): 2단 OOF
        comp = ["clip", "low", "d1", "d2", "d3"]; Z = np.stack([lg(oof[c]) for c in comp], 1); meta = np.zeros(len(y))
        for tr, te in V3.folds(grp, s):
            meta[te] = LogisticRegression(C=1.).fit(Z[tr], y[tr], sample_weight=w[tr]).predict_proba(Z[te])[:, 1]
        mm = LogisticRegression(C=1.).fit(Z, y, sample_weight=w); oof["d5"] = meta
        full["d5"] = mm.predict_proba(np.stack([lg(full[c]) for c in comp], 1))[:, 1]; red["d5"] = mm.predict_proba(np.stack([lg(red[c]) for c in comp], 1))[:, 1]
        res[s] = (oof, full, red)
    cands = {"D0": "d0", "D1": "d1s", "D2": "d2s", "D3": "d3s", "D4": "d4", "D5": "d5"}
    hard = np.isin(pt, ["composite", "full_style", "partial_gen", "partial_style"])
    print("\n== 결과 ==")
    for nm, key in [("기준 stack", "stack")] + list(cands.items()):
        print(f"[{nm}]")
        for s in SEEDS:
            oof, full, red = res[s]; p, o = full[key], oof[key]
            print(f"  s{s} 공개 {((p >= .5) == py).sum()}/21 logloss={log_loss(py, np.clip(p, 1e-4, 1 - 1e-4)):.3f} auc={roc_auc_score(py, p):.3f} 실제오탐={(p[py == 0] >= .5).sum()} 어려운12={(p[hard] >= .5).sum()} "
                  f"| 보조 {V3.bal_acc(o[aux], y[aux], t[aux]):.3f} | 실제오탐(OOF) {(o[kind == 'real'] >= .5).mean():.3f} | " +
                  " ".join(f"{x[:6]}={(o[t == x] >= .5).mean():.2f}" for x in V3.TYPES6) + f" | 빨강실제정답 {(red[key][yr == 0] < .5).mean():.2f}")
        print("    공개 개별: " + " ".join(f"{f[:4]}={q:.2f}" for f, q in zip(pf, res[0][1][key])))
    print(f"\n== 채택 규칙 (기준 stack, 후보 {K}개, 문턱 {THR}) ==")
    fa = grp[aux]; farms = np.array(sorted(set(fa))); ya, ta = y[aux], t[aux]
    for nm, key in cands.items():
        ok2, dirs, pw = [], [], []
        for s in SEEDS:
            oof, full, _ = res[s]; pb, pc = full["stack"], full[key]; ll = lambda q: log_loss(py, np.clip(q, 1e-4, 1 - 1e-4))
            ok2.append(bool(((pc >= .5) == py).sum() >= ((pb >= .5) == py).sum() and ll(pc) < ll(pb) and (pc[py == 0] >= .5).sum() <= (pb[py == 0] >= .5).sum()))
            ob, oc = oof["stack"][aux], oof[key][aux]; dirs.append(bool(V3.bal_acc(oc, ya, ta) > V3.bal_acc(ob, ya, ta)))
            rng = np.random.default_rng(100 + s); worse = valid = 0
            for _ in range(NB):
                pick = np.concatenate([np.where(fa == f)[0] for f in rng.choice(farms, len(farms))])
                if (ya[pick] == 0).sum() == 0 or (ya[pick] == 1).sum() == 0: continue
                valid += 1; worse += V3.bal_acc(oc[pick], ya[pick], ta[pick]) <= V3.bal_acc(ob[pick], ya[pick], ta[pick])
            pw.append(worse / max(valid, 1))
        print(f"  {nm}: 공개조건 {ok2} | 보조방향 {dirs} | p_worse {[round(x, 4) for x in pw]} → {'채택' if all(ok2) and all(dirs) and max(pw) < THR else '불채택'}")

if __name__ == "__main__":
    main()
