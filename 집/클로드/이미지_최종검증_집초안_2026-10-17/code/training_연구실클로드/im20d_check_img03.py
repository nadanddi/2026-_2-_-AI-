"""IM20d: 제출 후보 03(C3) 패키지 점검. 서버 실행.
1) ViT-B/16 순수 torch(vit.py) = open_clip 출력 확인  2) 공개 21장: 패키지 출력 vs 학습 특징 경로(B2·P16·지문 특징 + 내보낸 파라미터) 대조
3) 순서 무관  4) 시간  5) 전문가별 발동 현황
사용: python im20d_check_img03.py <package_dir>
"""
import sys, time, random, csv, numpy as np, torch
from pathlib import Path
PK = Path(sys.argv[1]); sys.path.insert(0, str(PK)); import inference as I  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
LOCAL = Path(__file__).resolve().parents[1] / "local"
PUB = Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플"

def main():
    m = I.load_model(str(PK / "model")); cfg = m["cfg"]; th = cfg["experts"]["theta"]; print("θ", th)
    import open_clip
    oc, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai", cache_dir=str(LOCAL / "weights/openclip"))
    oc = oc.visual.eval().to(m["dev"]).half(); x = torch.randn(4, 3, 224, 224, device=m["dev"]).half()
    with torch.no_grad(): print("ViT-B/16 vit.py vs open_clip 최대차", float((oc(x).float() - m["b16"](x).float()).abs().max()))
    lab = list(csv.DictReader(open(PUB / "sample_labels.csv"))); files = [r["file"] for r in lab]; py = np.array([int(r["label"]) for r in lab])
    paths = [str(PUB / "images" / f) for f in files]
    t = time.time(); pr = np.array(I.predict(m, paths)); print(f"공개 21장 추론 {time.time()-t:.1f}s, 정답 {int(((pr >= .5) == py).sum())}/21")
    # 학습 특징 경로 재계산
    zb = np.load(LOCAL / "feats/public21_B2.npz"); Fb = zb["feats"].astype(np.float32)[:, 1:]; Fb /= np.linalg.norm(Fb, axis=-1, keepdims=True) + 1e-8; eb = dict(zip(zb["files"], Fb.mean(1)))
    zp = np.load(LOCAL / "feats/public21_P16.npz"); ep = dict(zip(zp["files"], zp["feats"].astype(np.float32)))
    zl = np.load(LOCAL / "feats/public21_look.npz"); el = dict(zip(zl["files"], zl["feats"]))
    zlow = np.load(LOCAL / "feats/public21_low.npz"); elow = dict(zip(zlow["files"], zlow["feats"]))
    off = []
    for f in files:
        e = eb[f]; p = (float(I._lr(m["lr"], e[None])[0]) + float(I._hgb(m["hg"], elow[f][None].astype(np.float64))[0])) / 2; d1, d2, gap = el[f]
        r = cfg["l2_rule"]
        if r["d1_lo"] < d1 < r["d1_hi"] and gap > r["gap"] and p >= r["min_p"]: p = max(p, r["floor"])
        s = {"E_pg": float(I._lr(m["e_pg"], e[None])[0]), "E_ps": float(I._lr(m["e_ps"], e[None])[0]), "E_L": float(I._lr(m["e_l"], ep[f]).max())}
        for k, v in s.items(): p = max(p, I._g(v, th[k]))
        off.append((p, s))
    po = np.array([o[0] for o in off]); print("패키지 vs 특징경로 최대차", round(float(np.abs(po - pr).max()), 4))
    for f, q, (pp, s), yy in zip(files, pr, off, py):
        fired = [k for k, v in s.items() if v >= th[k]]
        print(f"   {f[:4]} y={yy} p={q:.3f} 전문가발동 {fired}")
    sh = paths[:]; random.Random(1).shuffle(sh); pr2 = dict(zip(sh, I.predict(m, sh))); print("순서 무관 최대차", max(abs(pr2[p] - q) for p, q in zip(paths, pr)))

if __name__ == "__main__":
    main()
