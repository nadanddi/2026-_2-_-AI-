"""IM65: 후보 M(결합 메타 모델, IM_결합메타_M_v1.md) 학습·평가. 서버(nadan). 입력 local/feats_h/*.npz(im43).
- 메타 학습 자료: h3 + h3j(흔들기 진짜) + h3s(작은 부분 생성) + h3g(SDXL 재그리기) + gptr·gptr_src 학습 절반. 형식 게이트 안·ok만.
- 전문가 점수는 촬영 묶음 5폴드 OOF(폴드마다 E_f·E_ps2·E_g7 = im44 train, E_gr = im59 방식 재학습).
- 메타 = 표준화 + 로지스틱(balanced), 입력 7개 logit(p6)·logit(ppre)·rc·logit(E_f/E_ps2/E_g7/E_gr). 게이트 안 = 메타, 밖 = p6.
- 배포 전문가 = 나단9(h_model_h2 + ip_model_v1). 비교 기준 = 나단9(같은 게이트).
사용: python im65_meta_train_eval.py [open_r5]   (open_r5 없으면 탐색 지표 G1~G4까지만, R5는 열지 않음)"""
import sys, io, csv, json, numpy as np
from pathlib import Path
from PIL import Image, JpegImagePlugin
from sklearn.model_selection import GroupKFold
OPEN = len(sys.argv) > 1 and sys.argv[1] == "open_r5"
sys.argv = [sys.argv[0], "no", ".0025", ".001", "h2", "jig"]
sys.path.insert(0, str(Path(__file__).resolve().parent)); import im44_h_train_eval as H  # noqa
from im43_h_feats import paths_of  # noqa
L = H.L; NB = 2000; TAG = "m_v1"
b = io.BytesIO(); Image.new("RGB", (64, 64), (120, 130, 140)).save(b, "JPEG", quality=92); REFQ = Image.open(io.BytesIO(b.getvalue())).quantization
def std(p):
    try: im = Image.open(p); return im.format == "JPEG" and JpegImagePlugin.get_sampling(im) == 2 and im.quantization == REFQ
    except Exception: return False
M8 = {}
for k, v in np.load(L / "preds/h_model_h2.npz").items(): a, c = k.split("__"); M8.setdefault(a, {})[c] = v
MG = {a.split("__")[1]: v for a, v in np.load(L / "preds/ip_model_v1.npz").items()}; T9 = json.load(open(L / "preds/ip_theta_v1.json")); assert abs(T9["theta"]["E_gr"] - .0311) < 1e-3
EXP = ("E_f", "E_ps2", "E_g7", "E_gr")
def man(sn): return list(csv.DictReader(open(L / sn / "manifest.csv", encoding="utf-8")))
lowemb = lambda F: np.c_[F["low"], F["emb"]]
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
def mfeat(G, sc): return np.c_[lg(G["p6"]), lg(G["ppre"]), G["rc"].astype(np.float32), *[lg(sc[k]) for k in EXP]]
def score9(G): return {**H.score(M8, G), "E_gr": H.lrp(MG, lowemb(G))}
def p9_of(G, sc): return H.apply(G["p6"], sc, T9["theta"], T9["theta_rc"], G["rc"])
acc = lambda p, y: .5 * (p[y == 0] < .5).mean() + .5 * (p[y == 1] >= .5).mean()

def build():
    parts = []
    for sn in ("im_set_h3", "im_set_h3j", "im_set_h3s", "im_set_h3g", "im_set_gptr", "im_set_gptr_src"):
        F = H.load(sn); rr = man(sn); assert len(rr) == len(F["files"]) and all(r["file"] == f for r, f in zip(rr, F["files"]))
        keep = np.array([r.get("split", "train") == "train" for r in rr]); s = np.array([std(p) for p in paths_of(sn)])
        print(f"  {sn}: {keep.sum()}장, 게이트 안 {s[keep].mean():.3f}, ok {F['ok'][keep].mean():.3f}", flush=True)
        keep &= s & F["ok"]
        for r in rr:
            if sn == "im_set_h3j": r["kind"] = "real_jig"
        parts.append((sn, {k: F[k][keep] for k in ("emb", "es", "low", "p6", "ppre", "rc")}, [r for r, kk in zip(rr, keep) if kk]))
    F = {k: np.concatenate([P[k] for _, P, _ in parts]) for k in ("emb", "es", "low", "p6", "ppre", "rc")}
    rows = [r for _, _, rr in parts for r in rr]; src = np.concatenate([[sn] * len(rr) for sn, _, rr in parts])
    return F, rows, src

def main():
    F, rows, src = build(); n = len(rows)
    y = np.array([int(r["label"]) for r in rows]); kd = np.array([r["kind"] for r in rows]); grp = np.array([r["group"] for r in rows])
    print("메타 학습", n, dict(zip(*np.unique(kd, return_counts=True))), "진짜", int((y == 0).sum()), "가짜", int(y.sum()), flush=True)
    # 폴드: 묶음 5폴드, GPT 학습 2묶음은 다른 폴드로 고정(im59와 같음)
    fold = np.full(n, -1)
    for f_, (_, te) in enumerate(GroupKFold(5).split(np.arange(n), groups=grp)): fold[te] = f_
    for f_, g_ in enumerate(sorted(set(grp[kd == "gpt_regen"]))): fold[grp == g_] = f_
    assert all(len(set(fold[grp == g_])) == 1 for g_ in set(grp))
    hmask = np.isin(src, ["im_set_h3", "im_set_h3j"])   # 나단8 전문가 학습 자료(같은 방식)
    rows_h = [dict(r, kind=("real" if r["kind"] == "real_jig" else r["kind"])) for r in rows]
    OV = {i: H.ov(L / "im_set_h3/mask" / rows[i]["file"].replace(".jpg", ".png")) for i in range(n) if src[i] == "im_set_h3" and kd[i] in ("partial_style", "partial_gen_v7")}
    grmask = np.isin(src, ["im_set_h3j", "im_set_gptr", "im_set_gptr_src"]) | ((src == "im_set_h3") & (kd == "real"))   # E_gr 학습 자료(im59와 같음)
    oof = {k: np.full(n, np.nan) for k in EXP}
    for f_ in range(5):
        tr, te = np.where(fold != f_)[0], np.where(fold == f_)[0]
        M = H.train(F, rows_h, tr[hmask[tr]], OV); sc = H.score(M, {k: F[k][te] for k in ("low", "emb", "es")})
        trg = tr[grmask[tr]]; mg = H.lrfit(lowemb(F)[trg], (kd[trg] == "gpt_regen").astype(int), C=.5); sc["E_gr"] = H.lrp(mg, lowemb({k: F[k][te] for k in ("low", "emb")}))
        for k in EXP: oof[k][te] = sc[k]
        print(f"  폴드 {f_}: 학습 {len(tr)} · 확인 {len(te)} (GPT 학습 {int((kd[tr] == 'gpt_regen').sum())})", flush=True)
    X = mfeat(F, oof)
    # 메타 교차 예측(참고) vs 나단9 규칙에 OOF 점수
    pm_oof = np.zeros(n)
    for f_ in range(5):
        tr, te = fold != f_, fold == f_; mm = H.lrfit(X[tr], y[tr], C=1.0); pm_oof[te] = H.lrp(mm, X[te])
    p9_oof = p9_of(F, oof)
    print(f"[학습 자료 안 OOF] 균형 정확도 나단9 규칙 {acc(p9_oof, y):.4f} → 메타 {acc(pm_oof, y):.4f}", flush=True)
    for kk in sorted(set(kd)):
        m = kd == kk; print(f"     {kk:16s} n={m.sum():4d} 정답률 {((p9_oof[m] >= .5) == (y[m] == 1)).mean():.3f} → {((pm_oof[m] >= .5) == (y[m] == 1)).mean():.3f}", flush=True)
    MM = H.lrfit(X, y, C=1.0); names = ["lg_p6", "lg_ppre", "rc", *[f"lg_{k}" for k in EXP]]
    print("메타 계수(표준화)", {nm: round(float(c), 3) for nm, c in zip(names, MM["coef"])}, "절편", round(float(MM["intercept"][0]), 3), flush=True)
    np.savez(L / f"preds/{TAG}.npz", **MM)

    C = {}
    def outputs(sn):
        G = H.load(sn); s = np.array([std(p) for p in paths_of(sn)]); sc = score9(G)
        return np.where(s, p9_of(G, sc), G["p6"]), np.where(s, H.lrp(MM, mfeat(G, sc)), G["p6"]), s
    def labels(sn):
        if sn == "pub21": return np.array([int(r["label"]) for r in csv.DictReader(open(Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플/sample_labels.csv"))]), None, None
        if sn == "jig500": return np.zeros(500, int), None, None
        rr = list(csv.DictReader(open(L / "sealed_c3" / sn[7:] / "manifest.csv", encoding="utf-8"))) if sn.startswith("sealed_") else man(sn)
        return np.array([int(r["label"]) for r in rr]), np.array([r["kind"] for r in rr]), np.array([r["group"] for r in rr])
    def rep(sn, sub=None):
        y_, k_, g_ = labels(sn); p9, pm, s = outputs(sn)
        if sub is not None: y_, k_, g_, p9, pm, s = y_[sub], k_[sub], g_[sub], p9[sub], pm[sub], s[sub]
        nfp = int(((p9 < .5) & (pm >= .5) & (y_ == 0)).sum()); lostfp = int(((p9 >= .5) & (pm < .5) & (y_ == 0)).sum())
        fx = int(((p9 < .5) & (pm >= .5) & (y_ == 1)).sum()); lost = int(((p9 >= .5) & (pm < .5) & (y_ == 1)).sum())
        st = f"[{sn}] n={len(y_)} 게이트 안 {s.mean():.2f} | 진짜: 새 오탐 {nfp} · 고친 오탐 {lostfp} | 가짜: 새로 잡음 {fx} · 놓치게 됨 {lost}"
        if (y_ == 0).any() and (y_ == 1).any(): st += f" | 균형 정확도 {acc(p9, y_):.4f} → {acc(pm, y_):.4f}"
        print(st, flush=True)
        if k_ is not None:
            for kk in sorted(set(k_)):
                m = k_ == kk; print(f"     {kk:18s} n={m.sum():4d} 정답률 {((p9[m] >= .5) == (y_[m] == 1)).mean():.3f} → {((pm[m] >= .5) == (y_[m] == 1)).mean():.3f}", flush=True)
        C[sn] = dict(y=y_, k=k_, g=g_, p9=p9, pm=pm, nfp=nfp, fx=fx, lost=lost)
    print("\n== 탐색 지표(R5 미개봉) ==", flush=True)
    for sn in ("im_set_r2", "im_set_r3", "im_set_r4", "sealed_real", "sealed_brown", "sealed_bright", "jig500", "pub21",
               "im_set_r3p_sharp", "im_set_r3p_png", "im_set_r3p_area", "im_set_r3p_webp", "im_set_r3p_c444",
               "im_set_gpt3", "im_set_gpt1", "im_set_sd2", "sealed_qwen", "sealed_v7", "im_set_newreal", "im_set_realg", "im_set_realn"): rep(sn)
    rep("im_set_gptr", np.array([r["split"] == "hold" for r in man("im_set_gptr")])); C["gptr_hold"] = C.pop("im_set_gptr")
    rep("im_set_gptr_src", np.array([r["split"] == "hold" for r in man("im_set_gptr_src")]))
    a_ = lambda sn, key: acc(C[sn][key], C[sn]["y"])
    g1 = a_("im_set_r2", "pm") >= a_("im_set_r2", "p9")
    g2 = a_("im_set_r3", "pm") >= a_("im_set_r3", "p9") and a_("im_set_r4", "pm") >= a_("im_set_r4", "p9")
    fr = lambda sn, key: (C[sn][key][C[sn]["y"] == 0] >= .5).mean()
    g3 = (sum(C[s]["nfp"] - int(((C[s]["p9"] >= .5) & (C[s]["pm"] < .5) & (C[s]["y"] == 0)).sum()) for s in ("sealed_real", "sealed_brown", "sealed_bright")) <= 2
          and fr("jig500", "pm") <= fr("jig500", "p9") + .01 and all(fr(f"im_set_r3p_{k}", "pm") <= fr(f"im_set_r3p_{k}", "p9") + .01 for k in ("sharp", "png", "area", "webp", "c444"))
          and int(((C["pub21"]["pm"] >= .5) == C["pub21"]["y"]).sum()) == 21)
    det = lambda sn, key: int((C[sn][key][C[sn]["y"] == 1] >= .5).sum())
    g4 = all(det(s, "pm") >= det(s, "p9") - 1 for s in ("im_set_gpt3", "im_set_gpt1", "im_set_sd2", "sealed_qwen", "sealed_v7", "gptr_hold"))
    print(f"\n[탐색 판정] G1 R2 {g1} · G2 R3·R4 {g2} · G3 진짜 오탐 {g3} · G4 학습 밖 생성물 {g4}", flush=True)
    if not OPEN or not (g1 and g2 and g3 and g4): print("M_TRAIN_DONE (R5 미개봉)" + ("" if OPEN else ""), flush=True); return
    print("\n== R5 1회 개봉 ==", flush=True); rep("im_set_r5"); d5 = C["im_set_r5"]; y5, g5 = d5["y"], d5["g"]
    gs = np.array(sorted(set(g5))); gi = {x: np.where(g5 == x)[0] for x in gs}; rng = np.random.default_rng(65); d = []
    for _ in range(NB):
        ix = np.concatenate([gi[x] for x in rng.choice(gs, len(gs))])
        if (y5[ix] == 0).any() and (y5[ix] == 1).any(): d.append(acc(d5["pm"][ix], y5[ix]) - acc(d5["p9"][ix], y5[ix]))
    pw = float((np.array(d) <= 0).mean()); fx = d5["fx"] - d5["lost"]; nfp = d5["nfp"] - int(((d5["p9"] >= .5) & (d5["pm"] < .5) & (y5 == 0)).sum())
    c = [acc(d5["pm"], y5) > acc(d5["p9"], y5), pw < .025, fx > nfp, nfp <= 3]
    print(f"[R5] 균형 정확도 {acc(d5['p9'], y5):.4f} → {acc(d5['pm'], y5):.4f}, p_worse {pw:.4f}, 순 고친 가짜 {fx} · 순 새 진짜 오탐 {nfp}", flush=True)
    print(f"== 판정(IM_결합메타_M_v1) == 상승 {c[0]} · p {c[1]} · 고침>오탐 {c[2]} · 오탐≤3 {c[3]} → {'채택 후보' if all(c) else '불채택'}", flush=True)
    print("M_EVAL_DONE", flush=True)

if __name__ == "__main__":
    main()
