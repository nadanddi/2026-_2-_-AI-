# -*- coding: utf-8 -*-
"""LK2 (2026-10-13 집 클로드; 사용자: 'LK2 진행').  사전 고정(커밋 후 실행).  링커 짝 정답 확률 모형.
정답 기준(학습 데이터만): truth.pkl 의 cert 연결(학습 기록 정답 EC·배지온 연속으로 정한 확실한 진짜 전날 a1) — 진단 달력 DM11 안 씀.
짝 = v15R 1순위 prev0.  ok = prev0 == a1 (대상이 cert 인 짝만).
특징(고정, 모두 규정 안 계산 가능): logit(p0), p0 − p1, p_null, ID 교차, 공기온 연속 |짝 23시 − 대상 0시|, 외기온 연속, 학습 달력(DMT3e) 날짜 일치(일치/불일치/모름 → 두 표시),
  같은 ID 앞 번호, HP2 일치(그 짝 표의 정답 보임 기준).  짝 EC 연속은 고EC 판단과 얽혀 제외.
모형: 표준화 + 로지스틱 회귀(C = 1), 거울 짝 표(v15r15_mirror, 정답 보임 = mirror_lab)에서 적합.
평가(EV 짝 표 v15r15_ev, 정답 보임 = 폴드 val∪del 숨김):  성공 = AUC(q) > AUC(p0)  AND  'q ≥ .8' 짝이 'p0 ≥ .8' 짝보다 정확도 ≥ 이고 짝 수 ≥.
출력: 모형 계수, EV 의 AUC·문턱별 정확도/개수, EV q 표(DS9 용) 저장.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
CR1SRC = open(os.path.join(HERE, "cr1_chain_gru_temp_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "cr1_chain_gru_temp_v1.py")
exec(CR1SRC[:CR1SRC.index("V = [\"in_temp\", \"in_hum\"")])   # hp2, mirror_lab, idx, TRAIN, FARM, NUM, N, rid
__file__ = os.path.join(HERE, "lk2_hop_correct_model_v1.py")
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import X as OUTX, S3  # noqa: E402
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CERT = {rid(int(b)): rid(int(a)) for a, b, c in zip(TL.a1, TL.b, TL.cert) if c}
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAYT = dict(zip(C.rec, C.day)); SEGT = dict(zip(C.rec, C.seg))
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
TIN = _X.pivot(index="rec", columns="h", values="in_temp").reindex(columns=range(24)); TOUT = _X.pivot(index="rec", columns="h", values="out_temp").reindex(columns=range(24))
def tdate(r, lab):
    i = idx[r]; vi = np.where(lab)[0]; vi = vi[vi != i]; m = vi[np.all(np.abs(OUTX[vi, 0] - OUTX[i, 0]) < 1e-6, axis=1)]
    ds = [(DAYT.get(rid(j)), SEGT.get(rid(j))) for j in m if rid(j) in DAYT]
    return pd.Series(ds).mode().iloc[0] if ds else None
def hopfeats(r, pv, p0, p1, pn, lab):
    td = tdate(r, lab); pdd = (DAYT.get(pv), SEGT.get(pv)) if (lab[idx[pv]] and pv in DAYT) else None
    dok = None if (td is None or pdd is None) else (pdd[1] == td[1] and pdd[0] == td[0] - 1)
    try: h2 = hp2(idx[r], lab)
    except Exception: h2 = None
    dTin = abs(TIN.at[pv, 23] - TIN.at[r, 0]); dTout = abs(TOUT.at[pv, 23] - TOUT.at[r, 0])
    return dict(lp0=np.log(np.clip(p0, 1e-4, 1 - 1e-4) / (1 - np.clip(p0, 1e-4, 1 - 1e-4))), margin=p0 - p1, p_null=pn, cross=float(pv[:3] != r[:3]),
                dTin=np.nan_to_num(dTin, nan=3.0), dTout=np.nan_to_num(dTout, nan=3.0), date_yes=float(dok is True), date_no=float(dok is False),
                same_id_earlier=float(pv[:3] == r[:3] and NUM[idx[pv]] < NUM[idx[r]]), hp2_agree=float(h2 == pv))
FEATS = ["lp0", "margin", "p_null", "cross", "dTin", "dTout", "date_yes", "date_no", "same_id_earlier", "hp2_agree"]
def table(PT, labf):
    rows = []
    for _, q in PT.iterrows():
        r, pv = q.rec, q.prev0
        if not isinstance(pv, str): continue
        f = hopfeats(r, pv, q.p0, q.p1, q.p_null, labf(q)); f.update(rec=r, prev=pv, p0=q.p0, ok=(int(pv == CERT[r]) if r in CERT else np.nan)); rows.append(f)
    return pd.DataFrame(rows)
PMr = PM.reset_index()
MT = table(PMr, lambda q: mirror_lab(q.rec)); print("mirror hops", len(MT), "| with cert truth", int(MT.ok.notna().sum()), "| acc", round(MT.ok.mean(), 3), flush=True)
PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_ev.csv"))
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hidEV = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
LABEV = {k: TRAIN & ~np.array([rid(i) in hs for i in range(N)]) for k, hs in hidEV.items()}
ET = table(PE, lambda q: LABEV[int(q.fold)]); ET["fold"] = PE.set_index("rec").loc[ET.rec, "fold"].values
print("EV hops", len(ET), "| with cert truth", int(ET.ok.notna().sum()), "| acc", round(ET.ok.mean(), 3), flush=True)
fit = MT[MT.ok.notna()]; sc = StandardScaler().fit(fit[FEATS]); LR = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(fit[FEATS]), fit.ok.astype(int))
print("\ncoefficients (standardized):", dict(zip(FEATS, np.round(LR.coef_[0], 3))))
ET["q"] = LR.predict_proba(sc.transform(ET[FEATS]))[:, 1]; MT["q"] = LR.predict_proba(sc.transform(MT[FEATS]))[:, 1]
E = ET[ET.ok.notna()]
a_q, a_p = roc_auc_score(E.ok, E.q), roc_auc_score(E.ok, E.p0); print(f"\nEV AUC: q {a_q:.3f} vs p0 {a_p:.3f}")
for t in (.5, .6, .7, .8, .9):
    print(f"  threshold {t}: q>=t n {int((E.q >= t).sum()):3d} acc {E[E.q >= t].ok.mean():.3f}  |  p0>=t n {int((E.p0 >= t).sum()):3d} acc {E[E.p0 >= t].ok.mean():.3f}")
nq, aq = int((E.q >= .8).sum()), E[E.q >= .8].ok.mean(); npp, ap = int((E.p0 >= .8).sum()), E[E.p0 >= .8].ok.mean()
ok = a_q > a_p and aq >= ap and nq >= npp
print(f"\nLK2 success (AUC q > p0 AND q>=.8 acc >= p0>=.8 acc AND count >=): {ok}  [AUC {a_q:.3f}/{a_p:.3f}, acc {aq:.3f}/{ap:.3f}, n {nq}/{npp}]")
print("\nDS8 culprit hops in EV:", ET[ET.rec.isin(["F47_126", "F13_160"])][["rec", "prev", "ok", "p0", "q"]].round(3).to_dict("records"))
ET.to_csv(os.path.join(HERE, "lk2_ev_hops_q_v1.csv"), index=False); MT.to_csv(os.path.join(HERE, "lk2_mirror_hops_q_v1.csv"), index=False)
import pickle; pickle.dump((sc, LR, FEATS), open(os.path.join(HERE, "lk2_model_v1.pkl"), "wb"))
