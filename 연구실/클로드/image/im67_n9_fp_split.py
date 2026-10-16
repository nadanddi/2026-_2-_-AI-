"""IM67: 진단(판정 아님) — im66에서 나단9 전문가 합집합 진짜 오탐 .0073의 출처 분해(최종 비평 제안). 서버(nadan).
① 미사용 진짜 3종(newreal·realg·realn) 게이트 안: 배포 전문가 + 나단9 θ  ② 학습 진짜(h3 진짜·gptr 짝 학습분) 게이트 안: 폴드 OOF 전문가(im66과 같은 폴드) + 나단9 θ.
전문가별 발동률과 최종 p9 ≥ .5 비율(p6 포함)을 함께 낸다."""
import sys, json, numpy as np
from pathlib import Path
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(Path(__file__).resolve().parent)); import im66_meta_v2 as M  # noqa
from sklearn.model_selection import GroupKFold
H = M.H; EXP = M.EXP; T9 = M.T9
def fires(sc, rc): return {k: sc[k] >= np.where(rc, T9["theta_rc"][k], T9["theta"][k]) for k in EXP}
print("== ① 미사용 진짜(배포 전문가) ==", flush=True); tot = []
for sn in M.UNSEEN:
    G = M.load_g(sn); m = G["std"] & G["ok"]; sc = M.score_with(M.M9, G); f = fires(sc, G["rc"]); un = np.any([f[k] for k in EXP], 0)
    p9 = M.p9_of(G, sc) if hasattr(M, "p9_of") else H.apply(G["p6"], sc, T9["theta"], T9["theta_rc"], G["rc"])
    print(f"  {sn}: 게이트 안 {m.sum()}/{len(m)} · 합집합 {un[m].mean():.4f} ({int(un[m].sum())}장) · " + " ".join(f"{k} {int(f[k][m].sum())}" for k in EXP) + f" | p6≥.5 {int((G['p6'][m] >= .5).sum())} · p9≥.5 {int((p9[m] >= .5).sum())}", flush=True)
    tot.append(un[m])
t = np.concatenate(tot); print(f"  미사용 합계 {t.mean():.4f} ({int(t.sum())}/{len(t)})", flush=True)
print("== ② 학습 진짜(폴드 OOF 전문가) ==", flush=True)
F, rows, src = M.build(); n = len(rows); y = np.array([int(r["label"]) for r in rows]); kd = np.array([r["kind"] for r in rows]); grp = np.array([r["group"] for r in rows]); gt = F["std"].astype(bool)
fold = np.full(n, -1)
for f_, (_, te) in enumerate(GroupKFold(5).split(np.arange(n), groups=grp)): fold[te] = f_
for f_, g_ in enumerate(sorted(set(grp[kd == "gpt_regen"]))): fold[grp == g_] = f_
hmask = np.isin(src, ["im_set_h3", "im_set_h3j"]); rows_h = [dict(r, kind=("real" if r["kind"] == "real_jig" else r["kind"])) for r in rows]
OV = {i: H.ov(H.L / "im_set_h3/mask" / rows[i]["file"].replace(".jpg", ".png")) for i in range(n) if src[i] == "im_set_h3" and kd[i] in ("partial_style", "partial_gen_v7")}
grmask = np.isin(src, ["im_set_h3j", "im_set_gptr", "im_set_gptr_src"]) | ((src == "im_set_h3") & (kd == "real"))
oof = {k: np.full(n, np.nan) for k in EXP}; dep = M.score_with(M.M9, F)
for f_ in range(5):
    tr, te = np.where(fold != f_)[0], np.where(fold == f_)[0]
    Mx = H.train(F, rows_h, tr[hmask[tr]], OV); trg = tr[grmask[tr]]; Mx["E_gr"] = H.lrfit(M.lowemb(F)[trg], (kd[trg] == "gpt_regen").astype(int), C=.5)
    sc = M.score_with(Mx, {k: F[k][te] for k in ("low", "emb", "es")})
    for k in EXP: oof[k][te] = sc[k]
r0 = (y == 0) & gt
for nm, sc in (("OOF", oof), ("배포(표본 안, 참고)", dep)):
    f = fires(sc, F["rc"]); un = np.any([f[k] for k in EXP], 0)
    print(f"  [{nm}] 학습 진짜 {r0.sum()} · 합집합 {un[r0].mean():.4f} ({int(un[r0].sum())}장) · " + " ".join(f"{k} {int(f[k][r0].sum())}" for k in EXP), flush=True)
print("N9SPLIT_DONE", flush=True)
