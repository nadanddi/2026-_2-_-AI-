"""IM59: 후보 I'(IM_GPT재그리기_Ip_v1.md) — 나단8 + E_gr(GPT 재그리기; 저수준38 + L/14 임베딩 로지스틱). 서버(nadan).
학습: im_set_h3 진짜 + im_set_h3j 흔들기 진짜 + im_set_gptr(split=train). 확인: gptr(split=hold), gpt3_s1, R4(open_r4일 때만), 진짜 오탐 세트들.
사용: python im59_ip_train_eval.py [open_r4]"""
import sys, io, csv, json, numpy as np
from pathlib import Path
from PIL import Image, JpegImagePlugin
from sklearn.model_selection import GroupKFold
OPEN = len(sys.argv) > 1 and sys.argv[1] == "open_r4"; PAIR = not (len(sys.argv) > 2 and sys.argv[2] == "nopair")   # 비평 반영: 짝 진짜 음성(기본 포함)
from scipy.stats import binomtest
sys.argv = [sys.argv[0], "no", ".0025", ".001", "h2", "jig"]
sys.path.insert(0, str(Path(__file__).resolve().parent)); import im44_h_train_eval as H  # noqa
from im43_h_feats import paths_of  # noqa
L = H.L; NB = 2000
b = io.BytesIO(); Image.new("RGB", (64, 64), (120, 130, 140)).save(b, "JPEG", quality=92); REFQ = Image.open(io.BytesIO(b.getvalue())).quantization
def std(p):
    try: im = Image.open(p); return im.format == "JPEG" and JpegImagePlugin.get_sampling(im) == 2 and im.quantization == REFQ
    except Exception: return False
z = np.load(L / "preds/h_model_h2.npz"); M8 = {}
for k, v in z.items(): a, c = k.split("__"); M8.setdefault(a, {})[c] = v
T8 = json.load(open(L / "preds/h_theta_h2.json"))
def man(sn): return list(csv.DictReader(open(L / sn / "manifest.csv", encoding="utf-8")))
def lowemb(F): return np.c_[F["low"], F["emb"]]
def patch_scores(m, F):
    es = F["es"].astype(np.float32); n = len(es); return H.lrp(m, es.reshape(n * 54, -1)).reshape(n, 54).max(1)

def main():
    parts = []
    for sn in ("im_set_h3", "im_set_h3j", "im_set_gptr") + (("im_set_gptr_src",) if PAIR else ()):
        F = H.load(sn); rr = man(sn); assert len(rr) == len(F["files"]) and all(r["file"] == f for r, f in zip(rr, F["files"]))
        keep = np.array([r["kind"] == "real" for r in rr]) if sn == "im_set_h3" else (np.array([r["split"] == "train" for r in rr]) if sn in ("im_set_gptr", "im_set_gptr_src") else np.ones(len(rr), bool))
        parts.append((sn, {k: v[keep] for k, v in F.items()}, [r for r, kk in zip(rr, keep) if kk]))
    F = {k: np.concatenate([P[k] for _, P, _ in parts]) for k in ("emb", "es", "low", "ok", "rc")}
    rows = [r for _, _, rr in parts for r in rr]
    kd = np.array(["gpt_regen" if r["kind"] == "gpt_regen" else "real" for r in rows]); grp = np.array([r["group"] for r in rows]); ok = F["ok"]
    print("I' 학습", {k: int((kd == k).sum()) for k in sorted(set(kd))}, "GPT 묶음", sorted(set(grp[kd == "gpt_regen"])), flush=True)
    train = lambda idx: {"E_gr": H.lrfit(lowemb(F)[idx], (kd[idx] == "gpt_regen").astype(int), C=.5)}
    score = lambda M, G: {"E_gr": H.lrp(M["E_gr"], lowemb(G))}
    allidx = np.where(ok)[0]; oof = np.full(len(rows), np.nan)
    fold = np.full(len(rows), -1)   # 실행 오류 수정(결과 전): GPT 학습 묶음 2개가 같은 폴드면 학습 양성 0 → 두 묶음을 다른 폴드로 고정
    for f_, (_, te) in enumerate(GroupKFold(5).split(allidx, groups=grp[allidx])): fold[allidx[te]] = f_
    for f_, g_ in enumerate(sorted(set(grp[kd == "gpt_regen"]))): fold[grp == g_] = f_
    for f_ in range(5):
        tr, te = allidx[fold[allidx] != f_], allidx[fold[allidx] == f_]
        if len(te): oof[te] = score(train(tr), {k: F[k][te] for k in ("emb", "low")})["E_gr"]
    MI = train(allidx)
    rm = (kd == "real") & ok; P = [oof[rm]]; prc = [F["rc"][rm]]; old = [np.zeros(rm.sum(), bool)]
    def old_fire(G, m):
        o = np.zeros(m.sum(), bool); sc = {"E_f": H.lrp(M8["E_f"], lowemb(G))[m], "E_ps2": patch_scores(M8["E_ps2"], G)[m], "E_g7": patch_scores(M8["E_g7"], G)[m]}
        for k in sc: o |= sc[k] >= np.where(G["rc"][m], T8["theta_rc"][k], T8["theta"][k])
        return o
    old[0] = old_fire({k: F[k] for k in ("emb", "es", "low", "rc")}, rm)
    for sn in ("im_set_newreal", "im_set_realg", "im_set_realn"):
        G = H.load(sn); m = G["ok"]; P.append(score(MI, G)["E_gr"][m]); prc.append(G["rc"][m]); old.append(old_fire(G, m))
    P, prc, old = np.concatenate(P), np.concatenate(prc), np.concatenate(old); print(f"기존 세 전문가 합집합 진짜 오탐 {old.mean():.4f} (N={len(prc)})", flush=True)
    for fl, flrc in ((.0025, .001), (.0015, .00075), (.001, .0005), (.0005, .0005)):
        TH = float(np.quantile(P, 1 - fl)); THrc = float(np.quantile(P[prc], 1 - flrc)); un = old | (P >= np.where(prc, THrc, TH))
        print(f"[θ 단계 {fl}/{flrc}] θ_gr {TH:.4f} θ_rc {THrc:.4f} | 합집합 진짜 오탐 {un.mean():.4f}", flush=True)
        if un.mean() <= .005: break
    else:
        print("== 합집합 예산 0.5% 초과 → 불채택 ==", flush=True); return
    m = (kd == "gpt_regen") & ok; print(f"   [OOF] GPT 학습분 θ 이상 {(oof[m] >= TH).mean():.3f} (n={m.sum()}; 학습 묶음 2개라 각 폴드는 다른 1묶음으로 학습) | 짝 음성 {PAIR}", flush=True)
    rb = np.random.default_rng(59); bt = [float(np.quantile(P[rb.integers(0, len(P), len(P))], 1 - fl)) for _ in range(200)]
    print(f"   [θ 부트스트랩 200] θ_gr 5/50/95% {np.percentile(bt, [5, 50, 95]).round(4).tolist()} → 그때 E_gr 단독 진짜 오탐 {[round(float((P >= t).mean()), 4) for t in np.percentile(bt, [5, 50, 95])]}", flush=True)
    # 참고: 5묶음 하나씩 빼기(GPT 150 전체, 짝 진짜 포함), 특징 기여(CLIP만/저수준만)
    GA = H.load("im_set_gptr"); ra = man("im_set_gptr"); ga = np.array([r["group"] for r in ra]); hm_all = np.array([r["split"] == "hold" for r in ra])
    SA = H.load("im_set_gptr_src"); base = (kd == "real") & ok & np.array([r.get("info", "") != "gptr_pair_real" for r in rows])
    for g in sorted(set(ga)):
        tr = ga != g; Xp = np.r_[lowemb(F)[base], lowemb(GA)[tr], lowemb(SA)[tr]]; yp = np.r_[np.zeros(base.sum()), np.ones(tr.sum()), np.zeros(tr.sum())]
        mm = H.lrfit(Xp, yp.astype(int), C=.5); print(f"   [묶음 빼기 {g}] n={(~tr).sum()} 탐지 {(H.lrp(mm, lowemb(GA)[~tr]) >= TH).mean():.3f}", flush=True)
    for nm, sl in (("CLIP만", slice(38, None)), ("저수준만", slice(0, 38))):
        idx = np.where(ok)[0]; mm = H.lrfit(lowemb(F)[idx][:, sl], (kd[idx] == "gpt_regen").astype(int), C=.5)
        q = np.concatenate([H.lrp(mm, lowemb(H.load(s_))[:, sl]) for s_ in ("im_set_newreal", "im_set_realg", "im_set_realn")]); t_ = float(np.quantile(q, 1 - fl))
        print(f"   [특징 {nm}] GT-hold 탐지 {(H.lrp(mm, lowemb(GA)[hm_all][:, sl]) >= t_).mean():.3f} (θ = 미사용 진짜 {len(q)}장 {1 - fl:.4f} 분위, 표본 안 아님)", flush=True)
    np.savez(L / "preds/ip_model_v1.npz", **{f"E_gr__{a}": v for a, v in MI["E_gr"].items()})
    json.dump(dict(theta={**T8["theta"], "E_gr": TH}, theta_rc={**T8["theta_rc"], "E_gr": THrc}), open(L / "preds/ip_theta_v1.json", "w"))
    thI = {**T8["theta"], "E_gr": TH}; thIrc = {**T8["theta_rc"], "E_gr": THrc}
    def outputs(sn):
        G = H.load(sn); s = np.array([std(p) for p in paths_of(sn)]); p6 = G["p6"]; rc = G["rc"]; s8 = H.score(M8, G)
        p8 = np.where(s, H.apply(p6, s8, T8["theta"], T8["theta_rc"], rc), p6); pI = np.where(s, H.apply(p6, {**s8, **score(MI, G)}, thI, thIrc, rc), p6); return p8, pI
    def labels(sn):
        if sn == "pub21": return np.array([int(r["label"]) for r in csv.DictReader(open(Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플/sample_labels.csv"))]), None, None
        if sn == "jig500": return np.zeros(500, int), None, None
        rr = list(csv.DictReader(open(L / "sealed_c3" / sn[7:] / "manifest.csv", encoding="utf-8"))) if sn.startswith("sealed_") else man(sn)
        return np.array([int(r["label"]) for r in rr]), np.array([r["kind"] for r in rr]), np.array([r["group"] for r in rr])
    C = {}
    def rep(sn):
        y, k, g = labels(sn); p8, pI = outputs(sn)
        nfp = int(((p8 < .5) & (pI >= .5) & (y == 0)).sum()); fx = int(((p8 < .5) & (pI >= .5) & (y == 1)).sum())
        print(f"[{sn}] n={len(y)} 새 진짜 오탐 {nfp} · 고친 가짜 {fx}", flush=True)
        if k is not None:
            for kk in sorted(set(k[y == 1])):
                mm = (k == kk) & (y == 1); print(f"     {kk:18s} n={mm.sum():4d} 나단8 {(p8[mm] >= .5).mean():.3f} → I' {(pI[mm] >= .5).mean():.3f}", flush=True)
        C[sn] = (y, k, g, p8, pI, nfp, fx)
    for sn in ("pub21", "jig500", "sealed_real", "sealed_brown", "sealed_bright", "im_set_r3", "im_set_gpt3", "im_set_gpt1", "im_set_sd2"): rep(sn)
    rr = man("im_set_gptr"); hm = np.array([r["split"] == "hold" for r in rr]); p8, pI = outputs("im_set_gptr"); gh = int((pI[hm] >= .5).sum() - (p8[hm] >= .5).sum())
    print(f"[GT-hold GPT 재그리기 n={hm.sum()}] 나단8 {(p8[hm] >= .5).mean():.3f} → I' {(pI[hm] >= .5).mean():.3f} (+{gh}장) | 학습분 {(p8[~hm] >= .5).mean():.3f} → {(pI[~hm] >= .5).mean():.3f}", flush=True)
    y, k, _, p8, pI = C["im_set_gpt3"][:5]; mm = k == "gpt3_s1"; a_ = int(((pI >= .5) & (p8 < .5) & mm).sum()); b_ = int(((pI < .5) & (p8 >= .5) & mm).sum())
    sp = binomtest(a_, a_ + b_, .5, alternative="greater").pvalue if a_ + b_ else 1.0
    print(f"[gpt3_s1 부호검정] I'만 맞힘 {a_} · 나단8만 맞힘 {b_} · p {sp:.4f}", flush=True)
    g_ok = (a_ - b_) >= 3 and sp < .1 and gh >= 5
    rep("im_set_gptr_src"); ys, _, _, p8s, pIs = C["im_set_gptr_src"][:5]; hs = np.array([r["split"] == "hold" for r in man("im_set_gptr_src")])
    pair_fp = int(((p8s < .5) & (pIs >= .5) & hs).sum()); print(f"[짝 진짜 확인 묶음 {hs.sum()}] 새 오탐 {pair_fp}", flush=True)
    y, k, _, p8, pI = C["im_set_r3"][:5]; r3_ok = (pI[y == 1] >= .5).sum() >= (p8[y == 1] >= .5).sum()
    pub_ok = int(((C["pub21"][4] >= .5) == C["pub21"][0]).sum()) == 21
    fp_ok = C["im_set_r3"][5] <= 2 and C["jig500"][5] <= 2 and sum(C[s][5] for s in ("sealed_real", "sealed_brown", "sealed_bright")) <= 2 and pair_fp <= 1
    print(f"\n[탐색 지표] GPT(GT-hold +5장 & gpt3_s1↑) {g_ok} · R3 가짜 {r3_ok} · 공개 {pub_ok} · 진짜 오탐(R3·jig·봉인) {fp_ok}", flush=True)
    if not OPEN: print("IP_TRAIN_DONE (R4 미개봉)", flush=True); return
    rep("im_set_r4"); y, k, g, p8, pI = C["im_set_r4"][:5]
    acc = lambda p: .5 * (p[y == 0] < .5).mean() + .5 * (p[y == 1] >= .5).mean()
    old7 = all((pI[(k == kk) & (y == 1)] >= .5).sum() >= (p8[(k == kk) & (y == 1)] >= .5).sum() for kk in sorted(set(k[y == 1])))
    print(f"[R4] 균형 정확도 나단8 {acc(p8):.4f} → I' {acc(pI):.4f}", flush=True)
    c = [g_ok, C["im_set_r4"][5] <= 2 and fp_ok, old7 and acc(pI) >= acc(p8), r3_ok and pub_ok]
    print(f"\n== 판정(IM_GPT재그리기_Ip_v1) == GPT {c[0]} · 진짜 오탐 {c[1]} · R4 무손해 {c[2]} · R3·공개 {c[3]} → {'채택 후보' if all(c) else '불채택'}", flush=True)
    print("IP_EVAL_DONE", flush=True)

if __name__ == "__main__":
    main()
