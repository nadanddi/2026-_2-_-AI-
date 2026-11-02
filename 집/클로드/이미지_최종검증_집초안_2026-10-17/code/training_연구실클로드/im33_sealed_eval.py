"""IM33: 봉인 확증 채점(IM_봉인확증_C3_v1.md) — 나단1·나단4 패키지 predict 그대로, 한 번만. 서버(nadan, GPU 0)."""
import sys, csv, importlib, numpy as np
from pathlib import Path
B = Path.home() / "storage/nadan/farmai_img"; S = B / "data/local/sealed_c3"; NB = 2000

def run(pk, paths):
    sys.path.insert(0, str(pk)); I = importlib.import_module("inference"); m = I.load_model(str(pk / "model")); p = np.array(I.predict(m, paths))
    sys.path.remove(str(pk)); del sys.modules["inference"]; return p

def main():
    R = []
    for sn in ("real", "qwen", "v7", "brown", "bright"):
        R += [dict(r, set=sn, path=str(S / sn / "img" / r["file"])) for r in csv.DictReader(open(S / sn / "manifest.csv", encoding="utf-8"))]
    paths = [r["path"] for r in R]; st = np.array([r["set"] for r in R]); kd = np.array([r["kind"] for r in R]); grp = np.array([r["group"] for r in R])
    pA = run(B / "submit_img01/팜모니_나단1", paths); pC = run(B / "submit_img04/팜모니_나단4", paths)
    np.savez(B / "data/local/preds/sealed_c3.npz", pA=pA, pC=pC, set=st, kind=kd, group=grp, path=np.array(paths))
    real, fake = st == "real", np.isin(st, ["qwen", "v7"])
    def est(p, idx=None):
        w = np.ones(len(p)) if idx is None else np.bincount(idx, minlength=len(p)).astype(float); h = (p >= .5).astype(float)
        fp = (h[real] * w[real]).sum() / max(w[real].sum(), 1e-9); det = (h[fake] * w[fake]).sum() / max(w[fake].sum(), 1e-9); return .5 * (1 - fp) + .5 * det, fp, det
    eA, fA, dA = est(pA); eC, fC, dC = est(pC)
    print(f"봉인 진짜 {real.sum()} · 봉인 가짜 {fake.sum()} (Qwen {(st == 'qwen').sum()}, 7판 {(st == 'v7').sum()})")
    print(f"나단1: 오탐 {fA:.4f} 탐지 {dA:.4f} 추정정확도 {eA:.4f}")
    print(f"나단4: 오탐 {fC:.4f} 탐지 {dC:.4f} 추정정확도 {eC:.4f}")
    use = np.where(real | fake)[0]; gs = np.array(sorted(set(grp[use]))); gi = {g: use[grp[use] == g] for g in gs}; rng = np.random.default_rng(7); bs = []
    for _ in range(NB):
        idx = np.concatenate([gi[g] for g in rng.choice(gs, len(gs))])
        if real[idx].any() and fake[idx].any(): bs.append(est(pC, idx)[0] - est(pA, idx)[0])
    pw = float((np.array(bs) <= 0).mean())
    c = [fC <= fA + .01, dC >= dA + .05, eC > eA and pw < .025]
    print(f"\n== 판정(IM_봉인확증_C3_v1.md) ==\n  오탐비열등 {c[0]} · 탐지+.05 {c[1]} · 추정정확도↑&p_worse {pw:.4f} {c[2]} → {'확증 통과' if all(c) else '확증 실패'}")
    print("== 진단")
    for k in sorted(set(kd[fake])):
        s = kd == k; print(f"  {k:22s} n={s.sum():3d} 탐지 나단1 {(pA[s] >= .5).mean():.2f} 나단4 {(pC[s] >= .5).mean():.2f}")
    for sn in ("brown", "bright"):
        s = st == sn; print(f"  특이 진짜 {sn:6s} n={s.sum():3d} 오탐 나단1 {(pA[s] >= .5).mean():.3f} 나단4 {(pC[s] >= .5).mean():.3f}")
    for nm, p in (("나단1", pA), ("나단4", pC)):
        print(f"  {nm} 경계선(.4~.6) 비율: 진짜 {((p[real] > .4) & (p[real] < .6)).mean():.3f} 가짜 {((p[fake] > .4) & (p[fake] < .6)).mean():.3f}")

if __name__ == "__main__":
    main()
