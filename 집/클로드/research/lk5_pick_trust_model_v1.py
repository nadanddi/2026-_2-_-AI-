# -*- coding: utf-8 -*-
"""LK5 (2026-10-15 집 클로드; 링커 정확도 높이기 2단계).  사전 고정(커밋 후 실행).  링커.
LK4: 후보 순위 모형 top-1 .716(v15R .695). 정답이 후보 안에 있으면 .93~.95 맞힘 → 틀림의 대부분은 '정답이 후보 밖(EV 23%)' 인데도 뭔가를 고르는 것,
  게다가 softmax 확신은 과대(q≥.8 정확도 .79).  앵커에 쓰는 건 '보이는 정답 기록' 짝뿐이고, 거기서 틀린 짝이 손해(LH2 F47_212 등).
→ LK5 = LK4 가 고른 짝을 믿을지 판정하는 신뢰 모형(LK2 를 LK4 짝에 맞게): 목표 = 그 짝이 진짜 전날인가.
  LK4 점수: 거울 표는 대상 단위 5겹 교차 적합(OOF), EV 표는 거울 전체로 적합(LK4 와 같음).
  특징(시각 h): s1(1등 점수), gap(1등 − 2등, 후보 1개면 5), q(softmax 최대), n_cand, 1등의 date_ok·date_bad·date_unk·fp_h·dTin·dTout·cross·v15p·v15rank·cand_vis,
    runner = 2등 지문/1등 지문 비(없으면 3), p_null(v15R 표).
  모형: 표준화 로지스틱(C 1), 거울 OOF 짝으로 적합 → EV 평가.
앵커 쓸 수 있는 짝 = 1등이 보이는 정답 기록.  문턱 t = 거울 OOF 에서 '보이는 짝 정확도 ≥ .90' 이 되는 가장 낮은 신뢰값(미리 정하는 규칙).
기준(같은 EV 대상): v15R prev0 이 보이는 정답(lab0) & p0 ≥ .8 (지금 v7·TNA8 계열과 같은 문턱).
성공(h = 0, 미리 고정): EV 에서 LK5 의 '앵커 짝' 정확도 ≥ 기준 정확도 − .02  AND  맞은 앵커 짝 수 > 기준의 맞은 수.
서술: h 3·6·9·12·23 같은 지표, 정확도-개수 곡선(신뢰 문턱 .5~.95).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u lk5_pick_trust_model_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
MT = pd.read_csv(os.path.join(HERE, "lk4_mirror_cands_v1.csv.gz")); ET = pd.read_csv(os.path.join(HERE, "lk4_ev_cands_v1.csv.gz"))
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv")).set_index("rec"); PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_ev.csv")).set_index("rec")
HS = (0, 3, 6, 9, 12, 23)
BASEF = ["v15p", "v15rank", "date_ok", "date_bad", "date_unk", "fp", "fp_rank", "fp_ratio", "dTin", "dTout", "cross", "same_id_earlier", "cand_vis", "hp2_agree", "n_cand"]
def F4(h): return [f"fp{h}" if f == "fp" else (f"fp_rank{h}" if f == "fp_rank" else (f"fp_ratio{h}" if f == "fp_ratio" else f)) for f in BASEF]
def score(train, test, h):
    F = F4(h); sc = StandardScaler().fit(train[F]); LR = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(train[F]), train.ok.astype(int)); return LR.decision_function(sc.transform(test[F]))
def picks(D, h, pn):
    rows = []
    for r, d in D.groupby("rec"):
        d = d.sort_values("s", ascending=False); t = d.iloc[0]; s2 = d.s.iloc[1] if len(d) > 1 else t.s - 5; f2 = d[f"fp{h}"].iloc[1] if len(d) > 1 else np.nan
        q = float(1 / np.exp(d.s - t.s).sum())
        rows.append(dict(rec=r, cand=t.cand, ok=t.ok, s1=t.s, gap=t.s - s2, q=q, n_cand=t.n_cand, date_ok=t.date_ok, date_bad=t.date_bad, date_unk=t.date_unk, fp=t[f"fp{h}"],
                         runner=(f2 / max(t[f"fp{h}"], 1e-3)) if np.isfinite(f2) else 3.0, dTin=t.dTin, dTout=t.dTout, cross=t.cross, v15p=t.v15p, v15rank=t.v15rank, vis=t.cand_vis, p_null=float(np.nan_to_num(pn.get(r, 0.0)))))
    return pd.DataFrame(rows)
TF = ["s1", "gap", "q", "n_cand", "date_ok", "date_bad", "date_unk", "fp", "runner", "dTin", "dTout", "cross", "v15p", "v15rank", "vis", "p_null"]
recs = np.array(sorted(MT.rec.unique())); fold = {r: k for k, r in zip(np.random.default_rng(0).permutation(len(recs)) % 5, recs)}
base = PE.reindex(sorted(ET.rec.unique())); CERT_ok = ET[ET.ok == 1].set_index("rec").cand.to_dict()
bm = (base.lab0.astype(bool) & (base.p0 >= .8)).values; bok = np.array([CERT_ok.get(r) == p for r, p in zip(base.index, base.prev0)])
print(f"baseline v15R (visible-labelled prev0 & p0>=.8) on EV: n {int(bm.sum())} correct {int(bok[bm].sum())} acc {bok[bm].mean():.3f}", flush=True)
verdict = None
for h in HS:
    MT["s"] = np.nan
    for k in range(5):
        tr = MT.rec.map(fold) != k; MT.loc[~tr, "s"] = score(MT[tr], MT[~tr], h)
    ET["s"] = score(MT, ET, h)
    Pm, Pe = picks(MT, h, PM.p_null.to_dict()), picks(ET, h, PE.p_null.to_dict())
    sc = StandardScaler().fit(Pm[TF]); LR = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Pm[TF]), Pm.ok.astype(int))
    Pm["t"] = LR.predict_proba(sc.transform(Pm[TF]))[:, 1]; Pe["t"] = LR.predict_proba(sc.transform(Pe[TF]))[:, 1]
    vm = Pm[Pm.vis == 1].sort_values("t", ascending=False); vm["cacc"] = vm.ok.expanding().mean()
    ok_t = vm[vm.cacc >= .90].t; thr = float(ok_t.min()) if len(ok_t) else 1.0
    ve = Pe[(Pe.vis == 1) & (Pe.t >= thr)]; n, c = len(ve), int(ve.ok.sum())
    line = f"h{h:2d}: threshold {thr:.3f} (mirror OOF visible acc>=.90) | EV anchor picks n {n} correct {c} acc {c / max(n, 1):.3f} | all-pick acc {Pe.ok.mean():.3f}"
    curve = " ; ".join(f"t>={t}: n {int(((Pe.vis == 1) & (Pe.t >= t)).sum())} acc {Pe[(Pe.vis == 1) & (Pe.t >= t)].ok.mean():.3f}" for t in (.5, .7, .8, .9))
    print(line + "\n      EV visible curve: " + curve, flush=True)
    if h == 0:
        verdict = (c / max(n, 1) >= bok[bm].mean() - .02) and (c > int(bok[bm].sum()))
        print("   trust coefficients:", dict(zip(TF, np.round(LR.coef_[0], 2))), flush=True)
        Pe.to_csv(os.path.join(HERE, "lk5_ev_picks_h0_v1.csv"), index=False)
print("\nLK5 success at h0:", verdict)
