# -*- coding: utf-8 -*-
"""HD1 (2026-10-14 집 클로드; 사용자: '고EC 반절, 고EC 아닌 날 반절로 새로 검증 만들어 정확도랑 포괄률 확인').  서술(채택 판정 아님), 실행 전 커밋.  EC.
목적: 고EC 날(하루 평균 sub_ec ≥ 1.0) 판별 능력만 따로 측정.  MT2 의 P2LOO AUC .95~.97 은 고EC 5일뿐이라 흔들림 → 학습 기록 전체로 다시.
판별기(모두 그 기록 자신의 입력만, 온실 ID 없음):
  R   = MT2 의 R(GRU 몸통 + 상태 머리 + 전문가 2 관문), 상태 확률 σ(s) 사용
  ET  = ExtraTrees 분류기(500그루, 잎 3, class_weight balanced) — 하루 0..h 누적/밤/낮 평균 14 입력(HR5 특징에서 base 뺀 것)
  각 판별기를 시각 h = 9(오전에 아는 것), 23(하루 전체) 두 시점으로.  R 의 h 시 확률은 0..h 입력만 본 출력(단방향).
OOF 배치(정답 있는 학습 기록 전부):
  B20 = 농장 x 번호 20 묶음을 한 폴드로(고EC 에피소드 통째로 숨김 → 엄격), 묶음을 10 폴드로 배정
  I10 = 번호 mod 10 폴드 + 같은 농장 ±1 숨김(가까운 날 ±2 는 학습에 남음 → P2LOO 와 비슷한 느슨한 배치)
  두 배치 모두 다른 농장 같은 번호 ±3 도 숨김(동시 운영 지문 공유 차단).
측정(사용자 요청):
  (1) 균형 검증: 고EC 전부 + 같은 수의 보통 날 무작위(농장 비율 맞춤) 200번 추출 평균 — 정확도(precision), 포괄률(recall), accuracy, AUC; 문턱 .3·.5·.7·.9
  (2) 실제 비율(전체 OOF, 고EC 약 9%) 에서의 정확도·포괄률 — 균형 검증은 정확도를 부풀리므로 같이 보고
  (3) 2차(번호 ≥ 179) 만, 놓친 고EC 기록·헛경보 보통 기록 목록(문턱 .5, h 23)
Run:  cd 집/클로드/research && PYTHONPATH="" python -u hd1_highec_detection_balanced_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "mt2_mtl_expert_informed_aux_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "mt2_mtl_expert_informed_aux_v1.py")
exec(src[:src.index("O = pd.read_csv")])
__file__ = os.path.join(HERE, "hd1_highec_detection_balanced_v1.py")
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
REC = list(LABREC); Yh = np.array([DM[idx[r]] >= 1.0 for r in REC]); farm = np.array([r[:3] for r in REC]); num = np.array([int(r[4:]) for r in REC])
OTH = {"F13": "F47", "F47": "F13"}; print("records", len(REC), "high", int(Yh.sum()), "pass-2 high", int((Yh & (num >= 179)).sum()), flush=True)
def dfeat(r, h):
    Z = RAW[idx[r]]
    with np.errstate(all="ignore"):
        cm = np.nanmean(Z[:h + 1], 0); nt = np.nanmean(Z[:min(h, 7) + 1], 0); dy = np.nanmean(Z[9:h + 1], 0) if h >= 9 else cm
    return np.concatenate([cm, nt, dy])
FE = {h: np.stack([dfeat(r, h) for r in REC]) for h in (9, 23)}
def schemes():
    blk = np.array([f"{f}_{n // 20}" for f, n in zip(farm, num)]); ub = sorted(set(blk)); rng = np.random.default_rng(7); asg = dict(zip(ub, rng.permutation(len(ub)) % 10))
    yield "B20", np.array([asg[b] for b in blk]), 0
    yield "I10", num % 10, 1
P = {}
for sc, fold, near in schemes():
    for k in range(10):
        te = np.where(fold == k)[0]
        hide = np.zeros(len(REC), bool); hide[te] = True
        for i in te:
            hide |= (farm == farm[i]) & (np.abs(num - num[i]) <= near); hide |= (farm == OTH[farm[i]]) & (np.abs(num - num[i]) <= 3)
        tr = [REC[i] for i in np.where(~hide)[0]]; tes = [REC[i] for i in te]
        _, s_, _ = fit_predict2(tr, tes, "R")
        for j, i in enumerate(te): P[(sc, "R", 9, i)] = s_[j][9]; P[(sc, "R", 23, i)] = s_[j][23]
        for h in (9, 23):
            X = FE[h]; Xt = np.where(np.isfinite(X[~hide]), X[~hide], np.nanmean(X[~hide], 0)); Xe = np.where(np.isfinite(X[te]), X[te], np.nanmean(X[~hide], 0))
            m = ExtraTreesClassifier(500, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=0).fit(Xt, Yh[~hide])
            for j, i in enumerate(te): P[(sc, "ET", h, i)] = m.predict_proba(Xe[j:j + 1])[0, 1]
        print(sc, "fold", k, "done", flush=True)
from sklearn.metrics import roc_auc_score
rows, rng = [], np.random.default_rng(20261014)
hiI = np.where(Yh)[0]; loI = {f: np.where(~Yh & (farm == f))[0] for f in ("F13", "F47")}
draws = [np.concatenate([hiI] + [rng.choice(loI[f], int((Yh & (farm == f)).sum()), replace=False) for f in ("F13", "F47")]) for _ in range(200)]
for sc in ("B20", "I10"):
    for mdl in ("R", "ET"):
        for h in (9, 23):
            p = np.array([P[(sc, mdl, h, i)] for i in range(len(REC))]); auc_all = roc_auc_score(Yh, p)
            for th in (.3, .5, .7, .9):
                q = p >= th; bal = [(precision_score(Yh[d], q[d], zero_division=0), recall_score(Yh[d], q[d]), accuracy_score(Yh[d], q[d])) for d in draws]
                p2 = num >= 179
                rows.append(dict(scheme=sc, model=mdl, hour=h, th=th, bal_precision=np.mean([b[0] for b in bal]), bal_recall=np.mean([b[1] for b in bal]), bal_acc=np.mean([b[2] for b in bal]),
                                 real_precision=precision_score(Yh, q, zero_division=0), real_recall=recall_score(Yh, q), false_alarms=int((q & ~Yh).sum()), missed=int((~q & Yh).sum()),
                                 p2_false_alarms=int((q & ~Yh & p2).sum()), p2_missed=int((~q & Yh & p2).sum()), AUC=auc_all))
R_ = pd.DataFrame(rows); pd.set_option("display.width", 250); print(R_.round(3).to_string(index=False))
for sc in ("B20", "I10"):
    for mdl in ("R", "ET"):
        p = np.array([P[(sc, mdl, 23, i)] for i in range(len(REC))])
        print(f"{sc} {mdl} h23 th.5 missed high:", [f"{REC[i]}({DM[idx[REC[i]]]:.2f},p{p[i]:.2f})" for i in np.where(Yh & (p < .5))[0]])
        print(f"{sc} {mdl} h23 th.5 false alarms:", [f"{REC[i]}({DM[idx[REC[i]]]:.2f},p{p[i]:.2f})" for i in np.where(~Yh & (p >= .5))[0]])
pd.to_pickle(P, os.path.join(env.LOCAL, "hd1_probs_v1.pkl"))
