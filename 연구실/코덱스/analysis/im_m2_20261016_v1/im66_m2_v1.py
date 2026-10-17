"""IM66: 후보 M v2(IM_결합메타_M_v2.md) — 나단9 + 다섯째 전문가 m(전문가 4개 logit의 로지스틱, OOF로 학습). 서버(nadan).
p_M = 게이트 안: max(p9, g(m; θ_m, θ_m_rc)) / 게이트 밖: p6.  사용: python im66_meta_v2.py [open_r5]"""
import sys, io, csv, json, numpy as np
from pathlib import Path
from PIL import Image, JpegImagePlugin
from sklearn.model_selection import GroupKFold
OPEN = len(sys.argv) > 1 and sys.argv[1] == "open_r5"
sys.argv = [sys.argv[0], "no", ".0025", ".001", "h2", "jig"]
sys.path.insert(0, str(Path.home() / "storage/nadan/farmai_img/data/image")); import im44_h_train_eval as H  # noqa
from im43_h_feats import paths_of  # noqa
L = H.L; NB = 2000; TAG = "m_v2"
import os
OUT = Path(os.environ["M2_STAGE_DIR"])
b = io.BytesIO(); Image.new("RGB", (64, 64), (120, 130, 140)).save(b, "JPEG", quality=92); REFQ = Image.open(io.BytesIO(b.getvalue())).quantization
def std(p):
    try: im = Image.open(p); return im.format == "JPEG" and JpegImagePlugin.get_sampling(im) == 2 and im.quantization == REFQ
    except Exception: return False
M8 = {}
for k, v in np.load(L / "preds/h_model_h2.npz").items(): a, c = k.split("__"); M8.setdefault(a, {})[c] = v
MG = {a.split("__")[1]: v for a, v in np.load(L / "preds/ip_model_v1.npz").items()}; T9 = json.load(open(L / "preds/ip_theta_v1.json")); assert abs(T9["theta"]["E_gr"] - .0311) < 1e-3
EXP = ("E_f", "E_ps2", "E_g7", "E_gr"); UNSEEN = ("im_set_newreal", "im_set_realg", "im_set_realn")
def man(sn): return list(csv.DictReader(open(L / sn / "manifest.csv", encoding="utf-8")))
lowemb = lambda F: np.c_[F["low"], F["emb"]]
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
def score_with(Mx, G): return {**H.score(Mx, G), "E_gr": H.lrp(Mx["E_gr"], lowemb(G))}
M9 = {**M8, "E_gr": MG}
def mX(sc): return np.c_[[lg(sc[k]) for k in EXP]].T
def fire9(sc, rc): return np.any([sc[k] >= np.where(rc, T9["theta_rc"][k], T9["theta"][k]) for k in EXP], 0)
acc = lambda p, y: .5 * (p[y == 0] < .5).mean() + .5 * (p[y == 1] >= .5).mean()
_cache = {}
def load_g(sn):
    if sn not in _cache: G = H.load(sn); G["std"] = np.array([std(p) for p in paths_of(sn)]); _cache[sn] = G
    return _cache[sn]

def build():
    parts = []
    for sn in ("im_set_h3", "im_set_h3j", "im_set_h3s", "im_set_h3g", "im_set_gptr", "im_set_gptr_src"):
        F = load_g(sn); rr = man(sn); assert len(rr) == len(F["files"]) and all(r["file"] == f for r, f in zip(rr, F["files"]))
        keep = np.array([r.get("split", "train") == "train" for r in rr])
        print(f"  {sn}: {keep.sum()}장, 게이트 안 {F['std'][keep].mean():.3f}, ok {F['ok'][keep].mean():.3f}", flush=True)
        keep &= F["ok"]   # 실행 오류 수정(결과 전): 흔들기 진짜는 전부 게이트 밖 → 전문가 학습엔 넣고(나단8과 같음) 메타 학습·θ 풀에는 게이트 안만
        parts.append((sn, {k: F[k][keep] for k in ("emb", "es", "low", "p6", "ppre", "rc", "std")}, [dict(r, kind=("real_jig" if sn == "im_set_h3j" else r["kind"])) for r, kk in zip(rr, keep) if kk]))
    F = {k: np.concatenate([P[k] for _, P, _ in parts]) for k in ("emb", "es", "low", "p6", "ppre", "rc", "std")}
    rows = [r for _, _, rr in parts for r in rr]; src = np.concatenate([[sn] * len(rr) for sn, _, rr in parts])
    return F, rows, src

def main():
    F, rows, src = build(); n = len(rows)
    y = np.array([int(r["label"]) for r in rows]); kd = np.array([r["kind"] for r in rows]); grp = np.array([r["group"] for r in rows]); sou = np.array([r["source"] for r in rows])
    gt = F["std"].astype(bool); print("전문가 학습 풀", n, dict(zip(*np.unique(kd, return_counts=True))), "| 메타 학습(게이트 안)", int(gt.sum()), dict(zip(*np.unique(kd[gt], return_counts=True))), flush=True)
    fold = np.full(n, -1)
    for f_, (_, te) in enumerate(GroupKFold(5).split(np.arange(n), groups=grp)): fold[te] = f_
    for f_, g_ in enumerate(sorted(set(grp[kd == "gpt_regen"]))): fold[grp == g_] = f_   # GPT 학습 2묶음 다른 폴드(묶음 전체가 같이 이동)
    assert all(len(set(fold[grp == g_])) == 1 for g_ in set(grp)) and all(len(set(fold[sou == s_])) == 1 for s_ in set(sou)), "묶음/원본이 폴드를 가로지름"
    print("폴드 크기", np.bincount(fold).tolist(), "GPT 폴드", sorted(set(fold[kd == "gpt_regen"])), flush=True)
    hmask = np.isin(src, ["im_set_h3", "im_set_h3j"]); rows_h = [dict(r, kind=("real" if r["kind"] == "real_jig" else r["kind"])) for r in rows]
    OV = {i: H.ov(L / "im_set_h3/mask" / rows[i]["file"].replace(".jpg", ".png")) for i in range(n) if src[i] == "im_set_h3" and kd[i] in ("partial_style", "partial_gen_v7")}
    grmask = np.isin(src, ["im_set_h3j", "im_set_gptr", "im_set_gptr_src"]) | ((src == "im_set_h3") & (kd == "real"))
    oof = {k: np.full(n, np.nan) for k in EXP}; FM = []
    for f_ in range(5):
        tr, te = np.where(fold != f_)[0], np.where(fold == f_)[0]
        M = H.train(F, rows_h, tr[hmask[tr]], OV); trg = tr[grmask[tr]]; M["E_gr"] = H.lrfit(lowemb(F)[trg], (kd[trg] == "gpt_regen").astype(int), C=.5); FM.append(M)
        sc = score_with(M, {k: F[k][te] for k in ("low", "emb", "es")})
        for k in EXP: oof[k][te] = sc[k]
        print(f"  폴드 {f_}: 학습 {len(tr)} · 확인 {len(te)} (GPT 학습 {int((kd[tr] == 'gpt_regen').sum())})", flush=True)
    # 분포 점검(고정 규칙): 미사용 진짜에서 배포 vs 폴드 평균 logit 분위
    U = [load_g(s) for s in UNSEEN]; um = [G["std"] & G["ok"] for G in U]
    dep = {k: np.concatenate([lg(score_with(M9, G)[k])[m] for G, m in zip(U, um)]) for k in EXP}
    fav = {k: np.concatenate([np.mean([lg(score_with(M, G)[k]) for M in FM], 0)[m] for G, m in zip(U, um)]) for k in EXP}
    shift = {k: max(abs(np.quantile(dep[k], q) - np.quantile(fav[k], q)) for q in (.99, .9975)) for k in EXP}
    FOLDAVG = max(shift.values()) > .3; print("분포 점검 logit 분위 차", {k: round(float(v), 3) for k, v in shift.items()}, "→ 메타 입력", "폴드 평균" if FOLDAVG else "배포 모델", flush=True)
    def meta_in(G): return mX({k: 1 / (1 + np.exp(-np.mean([lg(score_with(M, G)[k]) for M in FM], 0))) for k in EXP}) if FOLDAVG else mX(score_with(M9, G))
    X = mX(oof); m_oof = np.full(n, np.nan)
    for f_ in range(5):
        tr, te = (fold != f_) & gt, (fold == f_) & gt; m_oof[te] = H.lrp(H.lrfit(X[tr], y[tr], C=1.0), X[te])
    MM = H.lrfit(X[gt], y[gt], C=1.0); print("메타 계수(표준화)", {k: round(float(c), 3) for k, c in zip(EXP, MM["coef"])}, "절편", round(float(MM["intercept"][0]), 3), flush=True)
    # θ_m: 진짜 풀 = 학습 진짜 메타 OOF + 미사용 진짜 배포 m
    r0 = (y == 0) & gt; rpool_m = [m_oof[r0]] + [H.lrp(MM, meta_in(G))[m] for G, m in zip(U, um)]; rpool_rc = [F["rc"][r0]] + [G["rc"][m] for G, m in zip(U, um)]
    rpool_9 = [fire9({k: oof[k][r0] for k in EXP}, F["rc"][r0])] + [fire9(score_with(M9, G), G["rc"])[m] for G, m in zip(U, um)]
    P, PRC, O9 = np.concatenate(rpool_m), np.concatenate(rpool_rc), np.concatenate(rpool_9); print(f"진짜 풀 N={len(P)} (진짜 확정 {int(PRC.sum())}) · 나단9 전문가 합집합 오탐 {O9.mean():.4f}", flush=True)
    for fl, flrc in ((.0025, .001), (.0015, .00075), (.001, .0005), (.0005, .0005)):
        TH = float(np.quantile(P, 1 - fl)); THrc = float(np.quantile(P[PRC], 1 - flrc)); un = O9 | (P >= np.where(PRC, THrc, TH))
        print(f"[θ 단계 {fl}/{flrc}] θ_m {TH:.4f} θ_m_rc {THrc:.4f} | 합집합 진짜 오탐 {un.mean():.4f}", flush=True)
        if un.mean() <= float(O9.mean()) + .001: break
    else:
        print("== 합집합 예산 0.5% 초과 → 불채택 ==", flush=True); return
    gm_oof = gt & (np.nan_to_num(m_oof) >= np.where(F["rc"], THrc, TH)); f9_oof = gt & fire9(oof, F["rc"])
    print(f"[학습 자료 안 OOF, 게이트 안, 참고] 가짜 탐지(전문가만) 나단9 {f9_oof[gt & (y == 1)].mean():.4f} → +m {(f9_oof | gm_oof)[gt & (y == 1)].mean():.4f}", flush=True)
    for kk in sorted(set(kd[y == 1])):
        mk = (kd == kk) & gt; print(f"     {kk:16s} n={mk.sum():4d} 전문가 발동 {f9_oof[mk].mean():.3f} → {(f9_oof | gm_oof)[mk].mean():.3f}", flush=True)
    np.savez(OUT / f"{TAG}.npz", **MM, foldavg=np.array(FOLDAVG)); json.dump(dict(theta_m=TH, theta_m_rc=THrc, foldavg=bool(FOLDAVG)), open(OUT / f"{TAG}_theta.json", "w"))
    if FOLDAVG: np.savez(OUT / f"{TAG}_foldexp.npz", **{f"{f_}__{k}__{a}": v for f_, M in enumerate(FM) for k, d in M.items() for a, v in d.items()})

    C = {}
    def outputs(sn, sim=False):
        G = load_g(sn); sc = score_with(M9, G); m = H.lrp(MM, meta_in(G)); rc = np.zeros(len(m), bool) if sim else G["rc"]; base = G["ppre"] if sim else G["p6"]
        p9 = H.apply(base, sc, T9["theta"]) if sim else H.apply(base, sc, T9["theta"], T9["theta_rc"], rc)
        pm = np.maximum(p9, H.g(m, np.where(rc, THrc, TH))); return np.where(G["std"], p9, base), np.where(G["std"], pm, base), G["std"]
    def labels(sn):
        if sn == "pub21": return np.array([int(r["label"]) for r in csv.DictReader(open(Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플/sample_labels.csv"))]), None, None
        if sn == "jig500": return np.zeros(500, int), None, None
        rr = list(csv.DictReader(open(L / "sealed_c3" / sn[7:] / "manifest.csv", encoding="utf-8"))) if sn.startswith("sealed_") else man(sn)
        return np.array([int(r["label"]) for r in rr]), np.array([r["kind"] for r in rr]), np.array([r["group"] for r in rr])
    def rep(sn, sub=None, key=None):
        y_, k_, g_ = labels(sn); p9, pm, s = outputs(sn)
        if sub is not None: y_, p9, pm, s = y_[sub], p9[sub], pm[sub], s[sub]; k_ = k_[sub] if k_ is not None else None; g_ = g_[sub] if g_ is not None else None
        nfp = int(((p9 < .5) & (pm >= .5) & (y_ == 0)).sum()); fx = int(((p9 < .5) & (pm >= .5) & (y_ == 1)).sum())
        st = f"[{key or sn}] n={len(y_)} 게이트 안 {s.mean():.2f} | 새 진짜 오탐 {nfp} · 고친 가짜 {fx}"
        if (y_ == 0).any() and (y_ == 1).any(): st += f" | 균형 정확도 {acc(p9, y_):.4f} → {acc(pm, y_):.4f}"
        print(st, flush=True)
        if k_ is not None and (y_ == 1).any():
            for kk in sorted(set(k_[y_ == 1])):
                mk = (k_ == kk) & (y_ == 1); print(f"     {kk:18s} n={mk.sum():4d} 탐지 {(p9[mk] >= .5).mean():.3f} → {(pm[mk] >= .5).mean():.3f}", flush=True)
        C[key or sn] = dict(y=y_, k=k_, g=g_, p9=p9, pm=pm, nfp=nfp, fx=fx)
    print("\n== 탐색 지표(R5 미개봉) ==", flush=True)
    for sn in ("im_set_r2", "im_set_r3", "im_set_r4", "sealed_real", "sealed_brown", "sealed_bright", "jig500", "pub21", "im_set_r3p_sharp", "im_set_r3p_area",
               "im_set_r3p_png", "im_set_r3p_webp", "im_set_r3p_c444", "im_set_gpt3", "im_set_gpt1", "im_set_sd2", "sealed_qwen", "sealed_v7"): rep(sn)
    rep("im_set_gptr", np.array([r["split"] == "hold" for r in man("im_set_gptr")]), "gptr_hold"); rep("im_set_gptr_src", np.array([r["split"] == "hold" for r in man("im_set_gptr_src")]), "gptr_src_hold")
    a_ = lambda sn, key: acc(C[sn][key], C[sn]["y"]); fr = lambda sn, key: (C[sn][key][C[sn]["y"] == 0] >= .5).mean()
    g1 = a_("im_set_r2", "pm") >= a_("im_set_r2", "p9"); g2 = a_("im_set_r3", "pm") >= a_("im_set_r3", "p9") and a_("im_set_r4", "pm") >= a_("im_set_r4", "p9")
    g3 = (sum(C[s]["nfp"] for s in ("sealed_real", "sealed_brown", "sealed_bright")) <= 2 and fr("jig500", "pm") <= fr("jig500", "p9") + .01
          and all(fr(f"im_set_r3p_{k}", "pm") <= fr(f"im_set_r3p_{k}", "p9") + .01 for k in ("sharp", "area")) and int(((C["pub21"]["pm"] >= .5) == C["pub21"]["y"]).sum()) == 21 and C["gptr_src_hold"]["nfp"] <= 1)
    s9, sM = [], []
    for sn in UNSEEN + ("jig500", "sealed_real", "sealed_brown", "sealed_bright"):
        p9, pm, _ = outputs(sn, sim=True); yy = labels(sn)[0]; ok_ = load_g(sn)["ok"] & (yy == 0); s9.append(p9[ok_] >= .5); sM.append(pm[ok_] >= .5)
    s9, sM = np.concatenate(s9), np.concatenate(sM); g5 = sM.mean() <= s9.mean() + .005
    print(f"[G5 원본 제외 시뮬레이션] 진짜 {len(s9)}장 오탐 나단9 {s9.mean():.4f} → M {sM.mean():.4f}", flush=True)
    print(f"\n[탐색 판정] G1 R2 {g1} · G2 R3·R4 {g2} · G3 진짜 오탐 {g3} · G5 원본 제외 {g5}", flush=True)
    if not (g1 and g2 and g3 and g5): print("== 탐색 지표 불통과 → R5 열지 않음, M 불채택(후속 후보 없음) ==", flush=True); return
    if not OPEN: print("M_TRAIN_DONE (R5 미개봉)", flush=True); return
    print("\n== R5 1회 개봉 ==", flush=True); rep("im_set_r5"); d5 = C["im_set_r5"]; y5, g5_ = d5["y"], d5["g"]
    r5 = man("im_set_r5"); assert not ({r["group"] for r in r5} & set(grp)) and not ({r["source"] for r in r5} & set(sou)), "R5 ↔ 학습 묶음/원본 겹침"
    gs = np.array(sorted(set(g5_))); gi = {x: np.where(g5_ == x)[0] for x in gs}; rng = np.random.default_rng(66); d = []
    for _ in range(NB):
        ix = np.concatenate([gi[x] for x in rng.choice(gs, len(gs))])
        if (y5[ix] == 0).any() and (y5[ix] == 1).any(): d.append(acc(d5["pm"][ix], y5[ix]) - acc(d5["p9"][ix], y5[ix]))
    pw = float((np.array(d) <= 0).mean())
    c = [acc(d5["pm"], y5) > acc(d5["p9"], y5), pw < .025, d5["fx"] > d5["nfp"], d5["nfp"] <= 1]
    print(f"[R5] 균형 정확도 {acc(d5['p9'], y5):.4f} → {acc(d5['pm'], y5):.4f}, p_worse {pw:.4f}, 고친 가짜 {d5['fx']} · 새 진짜 오탐 {d5['nfp']}", flush=True)
    print(f"== 판정(IM_결합메타_M_v2) == 상승 {c[0]} · p {c[1]} · 고침>오탐 {c[2]} · 오탐≤1 {c[3]} → {'채택 후보' if all(c) else '불채택'}", flush=True)
    print("M_EVAL_DONE", flush=True)

if __name__ == "__main__":
    main()
