"""IM05h: im05f + R2 진단(추가 세트 실제의 농장별·배율 구간별 OOF 오탐). 판정: 공개 21장은 서술만(열람 상한 소진),
 채택 = 추가 세트 실제 오탐(OOF)이 시드 3개 모두 감소 AND 농장 군집 부트스트랩 p_worse<.025 AND 보조 가짜 탐지 평균이 0.01 넘게 떨어지지 않음.
원 설명: IM05f: 추가 자료 후보 일반 판정기(im05e 일반화). 사용: python im05f_eval_extra.py <추가세트> <어려운실제가중 HW> <이름>
 예) H1: im_set_h1 2.0 stackH  /  B: im_set_sd3 1.0 stackB
원본 설명: IM05e: 후보 H1 판정 — 평가 v4(im05d)와 같은 규칙. 기준과 후보가 학습하는 자료만 다름:
 기준 stack  : im_set_v4 + im_set_sd2 행으로만 학습 (h1 행 제외)
 후보 stackH : + im_set_h1(어려운 실제와 짝 가짜) 포함 학습, 어려운 실제 학습 가중 ×2(실제 총가중 .5는 유지)
 OOF·보조·공개 21장 예측은 같은 폴드(시드별)·같은 행에서 비교. 보조 행 선택도 둘에 공통.
 추가 진단: 우리 '어려운 실제'(h1 실제) 오탐률, 기존 자료 실제 오탐률.
"""
import sys, csv, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05d_eval_v4 as D, im05c_eval_v3 as V3  # noqa

SEEDS, NB = (0, 1, 2), 2000
EXTRA, HW, CN = sys.argv[1], float(sys.argv[2]), sys.argv[3]

def main():
    rows, X, isred = D.load(["im_set_v4", "im_set_sd2", EXTRA])
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Xk = {u: v[keep] for u, v in X.items()}
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); ish1 = np.array([r["_set"] == EXTRA for r in R])
    hardreal = ish1 & (kind == "real")
    w0 = V3.weights(kind[~ish1]); wb = np.zeros(len(R)); wb[~ish1] = w0
    wh = V3.weights(kind); tot = wh[y == 0].sum(); wh[hardreal] *= HW; wh[y == 0] *= tot / wh[y == 0].sum()
    lab, pf, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab]); pt = np.array([r["type"] for r in lab])
    print(f"행 {len(R)} (h1 {ish1.sum()}: 어려운 실제 {hardreal.sum()}) | 공개 21장 열람(B=3번째, H1=4번째)")
    S = Xk["clip"] / np.linalg.norm(Xk["clip"], axis=1, keepdims=True); PS = PX["clip"] / np.linalg.norm(PX["clip"], axis=1, keepdims=True)
    aux = []
    for x, n in [("real", 240)] + [(x, 40) for x in V3.TYPES6]:
        cand = np.where(t == x)[0]
        if len(cand): aux += list(cand[np.argsort(-(S[cand] @ PS[pt == x].T).max(1))[:n]])
    aux = np.array(aux); print("보조 행 중 h1:", ish1[aux].sum())
    res = {}
    for s in SEEDS:
        rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp)))
        bag = np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False))
        out = {}
        for nm, w, allow in [("stack", wb, ~ish1), (CN, wh, np.ones(len(R), bool))]:
            o = np.zeros(len(y)); pp = np.zeros(len(pf))
            for u in ("clip", "low"):
                ou = np.zeros(len(y))
                for tr, te in V3.folds(grp, s):
                    tr = tr[allow[tr]]; ou[te] = D.fit(u, Xk[u], y, w, tr, s).predict_proba(Xk[u][te])[:, 1]
                o += ou / 2; pp += D.fit(u, Xk[u], y, w, np.where(bag & allow)[0], s).predict_proba(PX[u])[:, 1] / 2
            out[nm] = (o, pp)
        res[s] = out
        for nm, (o, pp) in out.items():
            hard12 = np.isin(pt, ["composite", "full_style", "partial_gen", "partial_style"])
            print(f"  s{s} {nm:7s} 공개 {((pp >= .5) == py).sum()}/21 logloss={log_loss(py, np.clip(pp, 1e-4, 1 - 1e-4)):.3f} auc={roc_auc_score(py, pp):.3f} "
                  f"실제오탐={(pp[py == 0] >= .5).sum()} 어려운12={(pp[hard12] >= .5).sum()} | 보조 균형acc={V3.bal_acc(o[aux], y[aux], t[aux]):.3f} "
                  f"| 어려운실제 오탐={(o[hardreal] >= .5).mean():.3f} 기존실제 오탐={(o[(kind == 'real') & ~ish1] >= .5).mean():.3f} "
                  f"| 가짜탐지 " + " ".join(f"{x[:6]}={(o[t == x] >= .5).mean():.2f}" for x in V3.TYPES6))
    print("\n== 채택 규칙 (기준 stack, 후보 1개 → 문턱 .025) ==")
    fa = grp[aux]; farms = np.array(sorted(set(fa))); ok2, dirs, pw = [], [], []
    for s in SEEDS:
        (ob, pb), (oc, pc) = res[s]["stack"], res[s][CN]; ll = lambda q: log_loss(py, np.clip(q, 1e-4, 1 - 1e-4))
        ok2.append(bool(((pc >= .5) == py).sum() >= ((pb >= .5) == py).sum() and ll(pc) < ll(pb) and (pc[py == 0] >= .5).sum() <= (pb[py == 0] >= .5).sum()))
        ya, ta = y[aux], t[aux]; dirs.append(bool(V3.bal_acc(oc[aux], ya, ta) > V3.bal_acc(ob[aux], ya, ta)))
        rng = np.random.default_rng(100 + s); worse = valid = 0
        for _ in range(NB):
            pick = np.concatenate([np.where(fa == f)[0] for f in rng.choice(farms, len(farms))])
            if (ya[pick] == 0).sum() == 0 or (ya[pick] == 1).sum() == 0: continue
            valid += 1; worse += V3.bal_acc(oc[aux][pick], ya[pick], ta[pick]) <= V3.bal_acc(ob[aux][pick], ya[pick], ta[pick])
        pw.append(worse / max(valid, 1))
    print(f"  {CN}: 공개조건 {ok2} | 보조방향 {dirs} | p_worse {[round(x, 4) for x in pw]} → {'채택' if all(ok2) and all(dirs) and max(pw) < .025 else '불채택'}")

if __name__ == "__main__" and not (len(sys.argv) > 4 and sys.argv[4] == "r2"):
    main()

def r2_verdict():
    """R2 판정(사전 고정): 추가 세트 실제 OOF 오탐 감소(시드 3개) & 농장 군집 부트스트랩 p_worse<.025 & 보조 가짜 탐지 평균 하락 ≤ .01."""
    global main
    import im05d_eval_v4 as D2
    rows, X, isred = D.load(["im_set_v4", "im_set_sd2", EXTRA])
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Xk = {u: v[keep] for u, v in X.items()}
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); isx = np.array([r["_set"] == EXTRA for r in R]); xr = isx & (kind == "real")
    farm = np.array([r.get("farm_id", "") for r in R]); scale = np.array([float(r.get("scale") or "nan") for r in R])
    w0 = V3.weights(kind[~isx]); wb = np.zeros(len(R)); wb[~isx] = w0; wh = V3.weights(kind)
    lab, pf, PX = D.pub_feats(); pt = np.array([r["type"] for r in lab])
    S = Xk["clip"] / np.linalg.norm(Xk["clip"], axis=1, keepdims=True); PS = PX["clip"] / np.linalg.norm(PX["clip"], axis=1, keepdims=True)
    auxf = []
    for x in V3.TYPES6:
        cand = np.where(t == x)[0]
        if len(cand): auxf += list(cand[np.argsort(-(S[cand] @ PS[pt == x].T).max(1))[:40]])
    auxf = np.array(auxf); fx = farm[xr]; farms = np.array(sorted(set(fx)))
    print(f"\n== R2 판정 (추가 세트 실제 {xr.sum()}장, 농장 {len(farms)}곳) ==")
    allok = True
    for s in SEEDS:
        oo = {}
        for nm, w, allow in [("base", wb, ~isx), ("cand", wh, np.ones(len(R), bool))]:
            o = np.zeros(len(y))
            for u in ("clip", "low"):
                for tr, te in V3.folds(grp, s):
                    tr = tr[allow[tr]]; o[te] += D.fit(u, Xk[u], y, w, tr, s).predict_proba(Xk[u][te])[:, 1] / 2
            oo[nm] = o
        fb, fc = (oo["base"][xr] >= .5), (oo["cand"][xr] >= .5)
        rng = np.random.default_rng(300 + s); worse = 0
        for _ in range(NB):
            pick = np.concatenate([np.where(fx == f)[0] for f in rng.choice(farms, len(farms))]); worse += fc[pick].mean() >= fb[pick].mean()
        pw = worse / NB
        recb = np.mean([(oo["base"][auxf][t[auxf] == x] >= .5).mean() for x in V3.TYPES6]); recc = np.mean([(oo["cand"][auxf][t[auxf] == x] >= .5).mean() for x in V3.TYPES6])
        ok = fc.mean() < fb.mean() and pw < .025 and recc >= recb - .01; allok &= ok
        print(f"  s{s}: 추가 실제 오탐 {fb.mean():.3f} → {fc.mean():.3f} (p_worse {pw:.4f}) | 보조 가짜 탐지 평균 {recb:.3f} → {recc:.3f} | 기존 실제 오탐 "
              f"{(oo['base'][(kind == 'real') & ~isx] >= .5).mean():.3f} → {(oo['cand'][(kind == 'real') & ~isx] >= .5).mean():.3f} → {'통과' if ok else '실패'}")
        if s == 0:
            print("    농장별 추가 실제 오탐(기준→후보): " + " ".join(f"{f}={fb[fx == f].mean():.2f}→{fc[fx == f].mean():.2f}" for f in farms))
            sx = scale[xr]
            print("    배율 구간별: " + " ".join(f"[{lo},{hi})={fb[(sx >= lo) & (sx < hi)].mean():.2f}→{fc[(sx >= lo) & (sx < hi)].mean():.2f}" for lo, hi in [(0, .4), (.4, .5), (.5, .6), (.6, 9)]))
    print(f"  → R2 {'채택' if allok else '불채택'} (사전 고정 규칙)")

if __name__ == "__main__" and len(sys.argv) > 4 and sys.argv[4] == "r2":
    r2_verdict()
