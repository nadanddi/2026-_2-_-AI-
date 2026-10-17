"""IM20c: 이미지 제출 후보 03 = C3(유형별 전문가 위원회, IM_조합후보_C_v1.md) 최종 학습·내보내기. 서버 실행. (사용자 "c3패키지 그냥 한 번 만들어보자" 10-13)
구성: 기본 = 제출 01 모델(b2_lr·low_hgb 그대로) + R1 규칙(3.5<d1<20 · gap>8 · 규칙 전 p≥.25 → p≥.9)
      전문가 E_pg·E_ps = B2 조각평균 임베딩 → 표준화+로지스틱(진짜 대 해당 유형, 우리 가짜 + GPT v1·v2, 진짜/가짜 균형 가중)
      전문가 E_L = ViT-B/16(openai) 54조각(224, 간격 112) → 표준화+로지스틱(마스크 지도 조각 라벨, im05ad와 같은 규칙, 표본 시드 6) → 최댓값
θ_k = im05ad와 같은 교차적합(시드 6·7·8 × 5폴드, 기본이 진짜인 ev 진짜를 0.4% 뒤집는 값) 15개 평균
최종 출력 p = max(p_base, g_k(s_k)), g_k: s<θ → .5s/θ, 아니면 .5+.5(s−θ)/(1−θ)  (→ p≥.5 ⇔ 기본 가짜 또는 전문가 하나라도 θ 이상)
사용: python im20c_train_final_img03_C3.py <out_model_dir> <img01_model_dir>
"""
import sys, re, zlib, json, shutil, numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05z_eval_shortcut as Z  # noqa
from im05ab_eval_L2rule import rule  # noqa
from im05v_eval_patch import pfeat, pmask  # noqa
P, V3, D, bb, lookfeats = Z.P, Z.V3, Z.D, Z.bb, Z.lookfeats
LOCAL = D.LOCAL; SEEDS = (6, 7, 8); FLIP = .004; S2, HW = 224 * 224, 768 * 1024
SETS = ["im_set_v4", "im_set_sd2", "im_set_v5", "im_set_gpt1", "im_set_gpt2"]
OUT = Path(sys.argv[1]); IMG01 = Path(sys.argv[2])

def lr_export(m):
    sc, lg = m[0], m[-1]; return dict(mean=sc.mean_, scale=sc.scale_, coef=lg.coef_[0], intercept=lg.intercept_)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, Xd, isred = D.load(SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk = L[keep]; Xl = Xd["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); sets = np.array([r["_set"] for r in R]); N = len(R)
    gtr = np.isin(sets, ["im_set_gpt1", "im_set_gpt2"]); ev = ~gtr; realE = ev & (kind == "real")
    fz = {}
    for sn in SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    pf_ = {}
    for sn in SETS: pf_.update({(sn, k): v for k, v in pfeat(sn).items()})
    PF = np.stack([pf_[(r["_set"], r["file"])] for r in R]).astype(np.float32); NP = PF.shape[1]
    pm, garea = {}, {}
    for sn in SETS:
        pm.update({(sn, k): v for k, v in pmask(sn).items()}); p_ = LOCAL / "feats" / f"{sn}_pmask.npz"
        if p_.exists() and "area" in np.load(p_).files: zz = np.load(p_); garea.update({(sn, f): float(a) for f, a in zip(zz["files"], zz["area"])})
    gpart = gtr & np.isin(kind, ["gpt_pgen", "gpt_pstyle"])
    def make_ptr(seed):
        out = []
        for i, r in enumerate(R):
            key = (r["_set"], r["file"]); rng = np.random.default_rng(zlib.crc32(f"{seed}|{key}".encode()))
            if kind[i] == "real": out.append((i, rng.choice(NP, 12, replace=False), 0))
            elif kind[i] in ("gpt_gen", "gpt_fstyle"): out.append((i, rng.choice(NP, 12, replace=False), 1))
            elif key in pm and pm[key][1]:
                ov = pm[key][0]; area = garea.get(key) or float(re.search(r"area=([\d.]+)", r["info"]).group(1))
                pos = np.where((ov >= .25) | (ov * S2 >= .5 * area * HW))[0]
                if len(pos): out.append((i, pos, 1))
                if gpart[i]: continue
                zero = np.where(ov == 0)[0]
                if len(zero): out.append((i, rng.choice(zero, min(6, len(zero)), replace=False), 0))
        return out
    def fit_patch(ptr, tr):
        trs = set(tr.tolist()); Xs, ys = [], []
        for i, sel, lb in ptr:
            if i in trs: Xs.append(PF[i, sel]); ys.append(np.full(len(sel), lb))
        Xs = np.concatenate(Xs); ys = np.concatenate(ys); w = np.where(ys == 1, .5 / (ys == 1).sum(), .5 / (ys == 0).sum()) * len(ys)
        return make_pipeline(StandardScaler(), LogisticRegression(C=.05, max_iter=3000)).fit(Xs, ys, logisticregression__sample_weight=w)
    top1 = lambda mdl, F: mdl.predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(len(F), -1).max(1)
    wA = np.zeros(N); wA[ev] = V3.weights(kind[ev])
    def wexp(ty):
        sel = (kind == "real") | (t == ty); w = np.zeros(N); f = sel & (y == 1); r_ = sel & (y == 0); w[f] = .5 / f.sum(); w[r_] = .5 / r_.sum(); return sel, w * sel.sum()
    WE = {"E_pg": wexp("partial_gen"), "E_ps": wexp("partial_style")}
    TH = {k: [] for k in ("E_pg", "E_ps", "E_L")}
    for s in SEEDS:   # 교차적합 θ (im05ad와 같은 절차; 시험 세트가 없으므로 같은 폴드 배정을 위해 grp는 동일)
        ptr = make_ptr(s); oA = np.zeros(N); E = {k: np.zeros(N) for k in TH}; fold = np.zeros(N, int)
        for f, (tr, te) in enumerate(V3.folds(grp, s)):
            ta = tr[ev[tr]]; fold[te] = f
            oA[te] = (D.fit("clip", Xb, y, wA, ta, s).predict_proba(Xb[te])[:, 1] + D.fit("low", Xl, y, wA, ta, s).predict_proba(Xl[te])[:, 1]) / 2
            for k, (sel, w) in WE.items(): E[k][te] = D.fit("clip", Xb, y, w, tr[sel[tr]], s).predict_proba(Xb[te])[:, 1]
            E["E_L"][te] = top1(fit_patch(ptr, tr), PF[te])
        base = rule(oA, Lk, "R1")[0] >= .5
        for k in TH:
            for f in range(5):
                c = np.sort(E[k][realE & (fold != f) & ~base])[::-1]; kk = int(round(FLIP * (realE & (fold != f)).sum())); TH[k].append(float(c[kk - 1]) if kk >= 1 else 1.0)
        print(f"  θ 시드 {s}: " + " ".join(f"{k}={np.mean(v[-5:]):.5f}" for k, v in TH.items()), flush=True)
    theta = {k: float(np.mean(v)) for k, v in TH.items()}; print("θ 최종(15개 평균)", {k: round(v, 6) for k, v in theta.items()}, flush=True)
    idx = np.arange(N)
    for k, (sel, w) in WE.items(): np.savez(OUT / f"{k.lower()}_lr.npz", **lr_export(D.fit("clip", Xb, y, w, idx[sel], 0)))
    np.savez(OUT / "e_l_lr.npz", **lr_export(fit_patch(make_ptr(6), idx)))
    for f in ("b2_visual_fp16.safetensors", "fp_db.npz", "b2_lr.npz", "low_hgb.npz"): shutil.copyfile(IMG01 / f, OUT / f)
    import open_clip
    from safetensors.torch import save_file
    m, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai", cache_dir=str(LOCAL / "weights/openclip"))
    save_file({k: v.half().contiguous() for k, v in m.visual.state_dict().items()}, str(OUT / "b16_visual_fp16.safetensors"))
    cfg = dict(version="img03_C3", backbone="open_clip ViT-L-14-336 (openai) + ViT-B-16 (openai), visual towers fp16", view=336, grid=3, combine="mean(p_b2, p_low)",
               l2_rule=dict(d1_lo=3.5, d1_hi=20.0, gap=8.0, floor=0.9, min_p=0.25), experts=dict(theta=theta, flip_per_expert=FLIP, patch=224, stride=112),
               output="max(p_base, g_k(s_k)); g: s<θ → .5s/θ else .5+.5(s−θ)/(1−θ)", train_rows=int(N), gpt_rows=int(gtr.sum()))
    (OUT / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print("완료", sorted(x.name for x in OUT.iterdir()))

if __name__ == "__main__":
    main()
