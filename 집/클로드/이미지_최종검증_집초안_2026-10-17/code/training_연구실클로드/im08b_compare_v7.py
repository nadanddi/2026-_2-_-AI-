"""IM08b: 팀원 v7(리더보드 .77) vs 우리 B2 — 같은 보류 자료(im_set_r2: 농장 균형 실제 + 짝 가짜)에서 비교. 학습 없음(팀원), 우리는 v4+SD2(A)·+5판(B)로 학습해 r2를 예측.
팀원 v7: 제출 그대로(가중치 v5와 동일, dense 54패치·상위12). 코드는 압축 해제본 inference.py를 경로로 로드(검토 완료: 네트워크·설치 없음).
주의: 팀원 학습 자료(AI허브 1,475장)와 r2 실제가 겹칠 수 있음(팀원 쪽에 유리).
"""
import sys, csv, importlib.util, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05c_eval_v3 as V3  # noqa
V3.TYPE.update({"composite_v5b": "composite", "ai_gen_xl": "AI_gen", "full_style_xl": "full_style", "partial_style_xl": "partial_style", "partial_gen_xl": "partial_gen"})
import im05d_eval_v4 as D  # noqa
from im05k_backbones import bb  # noqa
LOCAL = D.LOCAL; V7 = LOCAL / "teammate_v7"

def v7_predict(paths):
    spec = importlib.util.spec_from_file_location("v7inf", V7 / "inference.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    model = m.load_model(str(V7 / "model")); return np.array(m.predict(model, [str(p) for p in paths]))

def main():
    rows, X, isred = D.load(["im_set_v4", "im_set_sd2", "im_set_v5", "im_set_r2"])
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Xl = X["low"][keep]
    fz = {}
    for sn in ("im_set_v4", "im_set_sd2", "im_set_v5", "im_set_r2"): fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    st = np.array([r["_set"] for r in R]); ev = (st == "im_set_r2") & (kind != "real_resampled")
    res = {}
    for nm, m in [("우리 B2+A", np.isin(st, ["im_set_v4", "im_set_sd2"])), ("우리 B2+B(5판)", np.isin(st, ["im_set_v4", "im_set_sd2", "im_set_v5"]))]:
        w = np.zeros(len(R)); w[m] = V3.weights(kind[m]); idx = np.where(m)[0]
        res[nm] = (D.fit("clip", Xb, y, w, idx, 0).predict_proba(Xb[ev])[:, 1] + D.fit("low", Xl, y, w, idx, 0).predict_proba(Xl[ev])[:, 1]) / 2
    res["팀원 v7"] = v7_predict([LOCAL / "im_set_r2" / "img" / R[i]["file"] for i in np.where(ev)[0]])
    ye, te = y[ev], t[ev]
    print(f"보류 자료 r2: 실제 {(ye == 0).sum()}, 가짜 {(ye == 1).sum()} (" + " ".join(f"{x}={(te == x).sum()}" for x in V3.TYPES6 if (te == x).any()) + ")")
    for nm, p in res.items():
        rec = {x: (p[te == x] >= .5).mean() for x in V3.TYPES6 if (te == x).any()}
        bal = .5 * (p[ye == 0] < .5).mean() + .5 * np.mean(list(rec.values()))
        print(f"  {nm:14s} 실제 정답률 {(p[ye == 0] < .5).mean():.3f} | 가짜 탐지 " + " ".join(f"{x[:6]}={v:.2f}" for x, v in rec.items()) + f" | 균형 정확도 {bal:.3f}")

if __name__ == "__main__":
    main()
