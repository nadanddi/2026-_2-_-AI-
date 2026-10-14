# -*- coding: utf-8 -*-
"""LK4 (2026-10-15 집 클로드; 사용자: C1 결과 후 '링커 정확도 높이는 방법 진행해봐').  사전 고정(커밋 후 실행).  링커(온도·EC 공통 복원 단계).
C1: 진짜 전날을 알면 온도 −12%, EC −40% → 대회 점수는 복원(링커) 정확도에 묶임.  지금까지: v15R 1순위 EV 정확도 .69(LK1), GP-RS .73(GP1),
  LK2 짝 정답 확률(1순위 짝 재채점, AUC .907), LKH1 시각별 지문(날짜 − 1 후보 안에서 0시 .84 → 9시 .94 → 23시 .97, 너그러운 기하).
  → 아직 안 한 것: 1순위 재채점이 아니라 '후보 전체' 를 시각별로 다시 순위 매기는 모형(v15R 후보 3 ∪ 학습 달력 날짜 − 1 후보, 특징 결합).
후보(대상 b, 보이는 정답 = lab): v15R 표 prev0·prev1·prev2 ∪ {보이는 정답 기록 중 DMT3e (날짜, 구간) = (대상 날짜 − 1, 같은 구간)}.  대상 날짜 = 0시 외기 일치 보이는 기록 최빈(LK2·LKH1 과 같음).
특징(후보마다, 시각 h 모형마다; 모두 규정 안 — 대상은 0..h 입력만, 후보는 앞 기록):
  v15p(v15R 확률, 목록 밖 0), v15rank(0·1·2, 밖 3), date_ok·date_bad·date_unk, fp_h = 운영 지문 8열 0..h 평균 |차|(LKH1), fp_rank, fp_ratio(후보 최소 대비),
  dTin·dTout(후보 23시 − 대상 0시), cross(ID 교차), same_id_earlier, cand_vis(보이는 정답 기록), hp2_agree, n_cand.
모형: 시각 h ∈ {0, 3, 6, 9, 12, 23} 마다 로지스틱(표준화, C 1), 거울 표(v15r15_mirror, mirror_lab) cert 대상의 후보 행으로 적합 → EV 표(v15r15_ev, 폴드 숨김) cert 대상으로 평가.
  선택 = 후보 중 점수 최대, 확신 q = softmax(로짓) 의 최대값.
정답 = truth.pkl cert(학습 정답 연속).
성공(미리 고정): h = 0 에서 EV top-1 정확도 > 같은 대상의 v15R prev0 정확도  AND  q ≥ .8 정확도 ≥ p0 ≥ .8 정확도 이고 개수 ≥.
서술: 시각별 정확도·q 문턱별 정확도/개수, 정답이 후보 안에 있는 비율(상한), 틀린 경우 단서, 보이는 정답 전날(앵커 쓸 수 있는) 정확도.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u lk4_candidate_ranker_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
CR1SRC = open(os.path.join(HERE, "cr1_chain_gru_temp_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "cr1_chain_gru_temp_v1.py")
exec(CR1SRC[:CR1SRC.index("V = [\"in_temp\", \"in_hum\"")])   # hp2, mirror_lab, idx, TRAIN, FARM, NUM, N, rid, PM
__file__ = os.path.join(HERE, "lk4_candidate_ranker_v1.py")
import warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import X as OUTX, S3  # noqa: E402
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CERT = {rid(int(b)): rid(int(a)) for a, b, c in zip(TL.a1, TL.b, TL.cert) if c}
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAYT = dict(zip(C.rec, C.day)); SEGT = dict(zip(C.rec, C.seg))
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
TIN = _X.pivot(index="rec", columns="h", values="in_temp").reindex(columns=range(24)); TOUT = _X.pivot(index="rec", columns="h", values="out_temp").reindex(columns=range(24))
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA[TRAIN], (0, 1))) / (np.nanstd(FPA[TRAIN], (0, 1)) + 1e-9)
HS = (0, 3, 6, 9, 12, 23)
FEATS = ["v15p", "v15rank", "date_ok", "date_bad", "date_unk", "fp", "fp_rank", "fp_ratio", "dTin", "dTout", "cross", "same_id_earlier", "cand_vis", "hp2_agree", "n_cand"]
def tdate(b, lab):
    vi = np.where(lab)[0]; vi = vi[vi != b]; m = vi[np.all(np.abs(OUTX[vi, 0] - OUTX[b, 0]) < 1e-6, axis=1)]
    ds = [(DAYT.get(rid(j)), SEGT.get(rid(j))) for j in m if rid(j) in DAYT]
    return pd.Series(ds).mode().iloc[0] if ds else None
def fpd(a, b, h):
    v = np.minimum(np.abs(FPA[a, :h + 1] - FPA[b, :h + 1]), 4); return float(np.nanmean(v)) if np.isfinite(v).any() else 2.0
def cand_rows(q, lab):
    r = q.rec; b = idx[r]; td = tdate(b, lab)
    try: h2 = hp2(b, lab)
    except Exception: h2 = None
    v15 = {}
    for j, (pv, pp) in enumerate(((q.prev0, q.p0), (q.prev1, q.p1), (q.prev2, q.p2))):
        if isinstance(pv, str) and pv != r and pv in idx: v15[pv] = (pp, j)
    dc = set() if td is None else {rid(i) for i in np.where(lab)[0] if i != b and DAYT.get(rid(i)) == td[0] - 1 and SEGT.get(rid(i)) == td[1]}
    cands = sorted(set(v15) | dc); rows = []
    for a_ in cands:
        a = idx[a_]; pdd = (DAYT.get(a_), SEGT.get(a_)) if (lab[a] and a_ in DAYT) else None
        dok = None if (td is None or pdd is None) else (pdd[1] == td[1] and pdd[0] == td[0] - 1)
        rows.append(dict(rec=r, cand=a_, v15p=v15.get(a_, (0.0, 3))[0] if np.isfinite(v15.get(a_, (0.0, 3))[0]) else 0.0, v15rank=v15.get(a_, (0.0, 3))[1],
                         date_ok=float(dok is True), date_bad=float(dok is False), date_unk=float(dok is None),
                         dTin=float(np.nan_to_num(abs(TIN.at[a_, 23] - TIN.at[r, 0]), nan=3.0)), dTout=float(np.nan_to_num(abs(TOUT.at[a_, 23] - TOUT.at[r, 0]), nan=3.0)),
                         cross=float(a_[:3] != r[:3]), same_id_earlier=float(a_[:3] == r[:3] and NUM[a] < NUM[b]), cand_vis=float(lab[a]), hp2_agree=float(h2 == a_), n_cand=len(cands),
                         **{f"fp{h}": fpd(a, b, h) for h in HS}, ok=float(a_ == CERT.get(r))))
    D = pd.DataFrame(rows)
    for h in HS:
        D[f"fp_rank{h}"] = D[f"fp{h}"].rank(method="min") - 1; D[f"fp_ratio{h}"] = D[f"fp{h}"] / max(D[f"fp{h}"].min(), 1e-3)
    return D
def build(PT, labf):
    out = []
    for _, q in PT.iterrows():
        if q.rec not in CERT: continue
        D = cand_rows(q, labf(q))
        if len(D): out.append(D)
    return pd.concat(out, ignore_index=True)
MT = build(PM.reset_index(), lambda q: mirror_lab(q.rec)); print("mirror: targets", MT.rec.nunique(), "candidate rows", len(MT), "| truth among candidates", round(MT.groupby("rec").ok.max().mean(), 3), flush=True)
PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_ev.csv"))
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hidEV = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
LABEV = {k: TRAIN & ~np.array([rid(i) in hs for i in range(N)]) for k, hs in hidEV.items()}
ET = build(PE, lambda q: LABEV[int(q.fold)]); print("EV: targets", ET.rec.nunique(), "candidate rows", len(ET), "| truth among candidates", round(ET.groupby("rec").ok.max().mean(), 3), flush=True)
v15 = PE.set_index("rec"); evt = sorted(ET.rec.unique())
base_acc = np.mean([v15.at[r, "prev0"] == CERT[r] for r in evt]); base_p = np.array([v15.at[r, "p0"] for r in evt]); base_ok = np.array([v15.at[r, "prev0"] == CERT[r] for r in evt])
print(f"\nv15R prev0 on same EV targets: acc {base_acc:.3f} (n {len(evt)}) | p0>=.8 n {int((base_p >= .8).sum())} acc {base_ok[base_p >= .8].mean():.3f}", flush=True)
res = {}
for h in HS:
    F = [f.replace("fp_rank", f"fp_rank{h}").replace("fp_ratio", f"fp_ratio{h}") if f in ("fp_rank", "fp_ratio") else (f"fp{h}" if f == "fp" else f) for f in FEATS]
    sc = StandardScaler().fit(MT[F]); LR = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(MT[F]), MT.ok.astype(int))
    ET["s"] = LR.decision_function(sc.transform(ET[F]))
    top = ET.loc[ET.groupby("rec").s.idxmax()].set_index("rec").reindex(evt)
    qmax = ET.groupby("rec").s.apply(lambda s: float(np.exp(s.max() - s.max()) / np.exp(s - s.max()).sum())).reindex(evt).values
    ok = top.ok.values.astype(bool); vis = top.cand_vis.values.astype(bool); res[h] = (ok, qmax)
    line = f"h{h:2d}: top-1 acc {ok.mean():.3f}"
    for t in (.5, .8, .9): line += f" | q>={t}: n {int((qmax >= t).sum())} acc {ok[qmax >= t].mean() if (qmax >= t).any() else float('nan'):.3f}"
    line += f" | visible-labelled picks n {int(vis.sum())} acc {ok[vis].mean():.3f}"
    print(line, flush=True)
    if h == 0: print("   h0 coefficients:", dict(zip(F, np.round(LR.coef_[0], 2))), flush=True)
ok0, q0 = res[0]; succ = ok0.mean() > base_acc and ((q0 >= .8).sum() >= (base_p >= .8).sum()) and (ok0[q0 >= .8].mean() >= base_ok[base_p >= .8].mean())
print(f"\nLK4 success at h0 (acc > v15R AND q>=.8 acc >= and count >=): {succ}  [acc {ok0.mean():.3f}/{base_acc:.3f}, q>=.8 n {int((q0 >= .8).sum())}/{int((base_p >= .8).sum())} acc {ok0[q0 >= .8].mean():.3f}/{base_ok[base_p >= .8].mean():.3f}]")
wrong = [r for r, o in zip(evt, ok0) if not o]; print("h0 wrong targets (sample):", wrong[:15])
ET.to_csv(os.path.join(HERE, "lk4_ev_cands_v1.csv.gz"), index=False); MT.to_csv(os.path.join(HERE, "lk4_mirror_cands_v1.csv.gz"), index=False)
