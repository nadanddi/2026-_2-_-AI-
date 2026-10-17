"""IM20b: 이미지 제출 후보 02 = G3c1 (IM_GPT학습_G3c_v1.md + 정정_v1) 최종 학습·내보내기. 서버 실행.
제출 01(im20)과 다른 점:
 - 학습 자료에 GPT v1·v2 250장 추가(im_set_gpt1·gpt2), 유형 안 GPT 몫 10%
 - 출력 끝에 단조 변환 g(τ=.5215 → .5): p<τ → .5p/τ, 아니면 .5+.5(p−τ)/(1−τ). τ = G3c1 교차적합 τ(시드 3·4·5 × 5폴드 15개) 평균(정정_v1 3항)
 - CLIP 시각부 가중치·지문 DB는 제출 01과 동일 파일(복사)
"""
import sys, json, shutil, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05x_eval_G3 as X  # noqa (TYPE 갱신 포함)
from im20_train_final_img01 import export_hgb, hgb_predict  # noqa
P, V3, D, bb = X.P, X.V3, X.D, X.bb
LOCAL = D.LOCAL; OUT = Path(sys.argv[1]); IMG01 = Path(sys.argv[2])
SETS = ["im_set_v4", "im_set_sd2", "im_set_v5", "im_set_gpt1", "im_set_gpt2"]
TAUS = [.516, .506, .523, .519, .506, .527, .530, .541, .540, .529, .518, .513, .518, .516, .520]   # logs/G3c_eval.log G3c1 행 그대로
TAU = float(np.mean(TAUS))
def g(p): return np.where(p < TAU, .5 * p / TAU, .5 + .5 * (p - TAU) / (1 - TAU))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, Xd, isred = D.load(SETS); keep = ~isred; R = [r for r, k in zip(rows, keep) if k]
    fz = {}
    for sn in SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R]); Xl = Xd["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    gtr = np.isin([r["_set"] for r in R], ["im_set_gpt1", "im_set_gpt2"]); w = V3.weights(kind)
    for x in P.GT:
        gg, o_ = gtr & (t == x), ~gtr & (t == x); tot = w[gg].sum() + w[o_].sum()
        if w[o_].sum() == 0 or gg.sum() == 0: continue
        w[gg] = tot * .10 / gg.sum(); w[o_] *= tot * .90 / w[o_].sum()
    idx = np.arange(len(R)); print(f"학습 행 {len(R)} (실제 {(y == 0).sum()}, 가짜 {y.sum()}, GPT {gtr.sum()}) τ={TAU:.4f}", flush=True)
    lr = D.fit("clip", Xb, y, w, idx, 0); hg = D.fit("low", Xl, y, w, idx, 0); sc, lg = lr[0], lr[-1]
    np.savez(OUT / "b2_lr.npz", mean=sc.mean_, scale=sc.scale_, coef=lg.coef_[0], intercept=lg.intercept_)
    ex = export_hgb(hg); np.savez(OUT / "low_hgb.npz", **ex)
    d_lr = np.abs(1 / (1 + np.exp(-(((Xb - sc.mean_) / sc.scale_) @ lg.coef_[0] + lg.intercept_[0]))) - lr.predict_proba(Xb)[:, 1]).max()
    d_hg = np.abs(hgb_predict(ex, Xl[:2000]) - hg.predict_proba(Xl[:2000])[:, 1]).max()
    print(f"내보내기 검산: LR 최대차 {d_lr:.2e}, HGB 최대차 {d_hg:.2e}", flush=True); assert d_lr < 1e-6 and d_hg < 1e-6
    for f in ("b2_visual_fp16.safetensors", "fp_db.npz"): shutil.copyfile(IMG01 / f, OUT / f)
    cfg = dict(version="img02_G3c1", backbone="open_clip ViT-L-14-336 (openai), visual tower fp16", view=336, grid=3, combine="mean(p_b2, p_low)",
               l2_rule=dict(d1_lo=3.5, d1_hi=20.0, gap=8.0, floor=0.9), tau=TAU, transform="p<tau: .5*p/tau ; else .5+.5*(p-tau)/(1-tau)",
               train_rows=int(len(R)), gpt_rows=int(gtr.sum()), gpt_share=.10)
    (OUT / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    lab, pf, PX = D.pub_feats(); pz = bb("public21", "B2"); Pb = np.stack([pz[f] for f in pf])
    p = (lr.predict_proba(Pb)[:, 1] + hg.predict_proba(PX["low"])[:, 1]) / 2
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pf])
    hit = (PL[:, 0] > 3.5) & (PL[:, 0] < 20) & (PL[:, 2] > 8); pr = g(np.where(hit, np.maximum(p, .9), p))
    np.savez(OUT.parent / "offline_public21.npz", files=np.array(pf), p_stack=p, p_final=pr)
    py = np.array([int(r["label"]) for r in lab]); print(f"공개 21장(오프라인 경로) 정답 {int(((pr >= .5) == py).sum())}/21: " + " ".join(f"{f[:4]}={q:.3f}" for f, q in zip(pf, pr)))
    print("완료", sorted(x.name for x in OUT.iterdir()))

if __name__ == "__main__":
    main()
