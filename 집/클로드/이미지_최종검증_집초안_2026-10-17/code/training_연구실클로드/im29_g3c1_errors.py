"""IM29: G3c1 오답 진단(판정 없음). 시드 3, 교차적합 τ(im05aa와 동일 규칙). A(.5)와 나란히.
출력: local/preds/g3c1_s3.npz, logs 요약(유형별 오답 수·오답 확률 분포·놓친 v3 특징), 오답 사진 모음 local/../q_err3/*.jpg
"""
import sys, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05z_eval_shortcut as Z  # noqa
P, V3, D, bb, lookfeats = Z.P, Z.V3, Z.D, Z.bb, Z.lookfeats
LOCAL = D.LOCAL; SETS = Z.SETS; S = 3; OUT = LOCAL.parents[1] / "q_err3"

def main():
    OUT.mkdir(exist_ok=True)
    rows, Xd, isred = D.load(SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk = L[keep]; Xl = Xd["low"][keep]; Xr = Xd["lowA"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); sets = np.array([r["_set"] for r in R]); realm = kind == "real"; N = len(R)
    gtr = np.isin(sets, ["im_set_gpt1", "im_set_gpt2"]); v3 = sets == "im_set_gpt3"; rn = sets == "im_set_realn"
    test = np.isin(sets, ["im_set_gpt3", "im_set_gpt3r", "im_set_realn", "im_set_realg"]); ev = ~gtr & ~test
    meth = np.array([r["info"].split(";")[1].split("_")[2] if r["_set"] == "im_set_gpt3" else "" for r in R])
    fz = {}
    for sn in SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    pr = {}
    for arm in ("A", "G3c1"):
        m = ev.copy() if arm == "A" else (ev | gtr); w = np.zeros(N); w[m] = V3.weights(kind[m])
        if arm == "G3c1":
            for x in P.GT:
                g, o_ = gtr & (t == x), ev & (t == x); tot = w[g].sum() + w[o_].sum()
                if w[o_].sum() == 0 or g.sum() == 0: continue
                w[g] = tot * .10 / g.sum(); w[o_] *= tot * .90 / w[o_].sum()
        o = np.zeros(N); fold = np.zeros(N, int)
        for f, (tr, te) in enumerate(V3.folds(grp, S)):
            tr = tr[m[tr]]; o[te] = (D.fit("clip", Xb, y, w, tr, S).predict_proba(Xb[te])[:, 1] + D.fit("low", Xl, y, w, tr, S).predict_proba(Xl[te])[:, 1]) / 2; fold[te] = f
        pr[arm] = (P.rule(o, Lk), fold)
    a, fold = pr["A"]; g_, _ = pr["G3c1"]; ths = []
    for f in range(5):
        oth = realm & (fold != f); ths.append(float(np.quantile(g_[oth], 1 - (a[oth] >= .5).mean())))
    th = np.array(ths)[fold]; hA = a >= .5; hG = g_ >= th; marg = g_ - th
    np.savez(LOCAL / "preds" / "g3c1_s3.npz", files=np.array([f"{r['_set']}/{r['file']}" for r in R]), pA=a, pG=g_, th=th, y=y, kind=kind)
    print(f"τ 폴드 {[round(x, 3) for x in ths]}")
    def line(nm, sel, pos):
        wa = (hA[sel] != pos).sum(); wg = (hG[sel] != pos).sum(); mm = marg[sel & (hG != pos)]
        print(f"  {nm:22s} n={sel.sum():5d} 오답 A {wa:4d} ({wa / sel.sum():.3f}) G3c1 {wg:4d} ({wg / sel.sum():.3f})"
              + (f" | G3c1 오답의 기준선까지 거리 중앙값 {np.median(np.abs(mm)):.3f}, 0.1 이내 {np.mean(np.abs(mm) < .1):.2f}" if len(mm) else ""))
    print("== 유형별 오답 (진짜는 가짜로 본 수, 가짜는 놓친 수)")
    line("진짜(OOF)", realm, False); line("진짜 새 300(realN)", rn, False)
    for x in V3.TYPES6: line(f"우리 가짜 {x}", ev & (t == x), True)
    line("v3 1단계(재그리기)", v3 & (kind == "gpt3_s1"), True); line("v3 A최종", v3 & (meth == "A") & (kind == "gpt3_pgen"), True)
    line("v3 B최종", v3 & (meth == "B"), True); line("v3 부분스타일(C)", v3 & (kind == "gpt3_pstyle"), True)
    print("== 바뀐 것: A는 맞고 G3c1은 틀림 / A는 틀리고 G3c1은 맞음")
    for nm, sel, pos in [("진짜", realm | rn, False), ("우리 가짜", ev & (y == 1), True), ("v3", v3, True)]:
        print(f"  {nm}: 새로 틀림 {((hA == pos) & (hG != pos) & sel).sum()} / 새로 맞음 {((hA != pos) & (hG == pos) & sel).sum()}")
    # 놓친 v3 특징: 원본대조 거리, 밝기·채도·선명도(저수준 일부)
    miss = v3 & ~hG; got = v3 & hG
    names = {"잡음잔차sd": 0, "라플라시안(선명도)": 3, "채도평균": 5, "밝기평균": 7, "밝기sd": 8}
    print("== 놓친 v3 vs 잡은 v3 (중앙값)")
    print(f"  원본대조 d1: 놓침 {np.median(Lk[miss, 0]):.1f} / 잡음 {np.median(Lk[got, 0]):.1f}")
    for nm, i in names.items(): print(f"  {nm}: 놓침 {np.median(Xl[miss, i]):.3f} / 잡음 {np.median(Xl[got, i]):.3f} / 진짜 {np.median(Xl[realm, i]):.3f}")
    def sheet(sel, name, n=12):
        idx = np.where(sel)[0]; idx = idx[np.argsort(marg[idx])][:n] if name.startswith("놓친") else idx[np.argsort(-marg[idx])][:n]
        tiles = []
        for i in idx:
            r = R[i]; im = Image.open(LOCAL / r["_set"] / "img" / r["file"]).convert("RGB")
            im = im.resize((192, 256)) if im.height >= im.width else im.resize((256, 192)).crop((32, 0, 224, 192)).resize((192, 256)); tiles.append(np.asarray(im))
        if not tiles: return
        rows_ = [np.concatenate(tiles[j:j + 6] + [np.zeros_like(tiles[0])] * (6 - len(tiles[j:j + 6])), 1) for j in range(0, len(tiles), 6)]
        Image.fromarray(np.concatenate(rows_, 0)).save(OUT / f"{name}.jpg", quality=85)
        print(f"  [{name}] " + " ".join(f"{R[i]['file'][:18]}({g_[i]:.2f}/τ{th[i]:.2f})" for i in idx))
    print("== 오답 모음(가장 크게 틀린 순)")
    sheet(miss, "놓친_v3"); sheet((realm | rn) & hG, "진짜오탐_G3c1"); sheet(ev & (y == 1) & hA & ~hG, "새로놓친_우리가짜")

if __name__ == "__main__":
    main()
