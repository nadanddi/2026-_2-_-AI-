# -*- coding: utf-8 -*-
"""VC1 (2026-10-13 집 클로드; 사용자: '검증기가 제대로 검증하는 게 맞는지 확인').  서술·감사(판정 아님).  EC, HMM 갈래(DS7·DS8) 기준.
질문: DS7·DS8 판정에 쓴 검증기(DIAG10·P2LOO·EL1·EV)의 시험 기록이 '평가 기록이 놓인 상황' 과 같은가?
기록 단위 상황 지표(각 검증기는 그 폴드의 숨김 기준, 평가는 학습 기록만 보임):
  pass2 = 번호 ≥ 179;  predH = 예측 하루 평균 ≥ 1.0;  prev_vis = v15R 1순위 짝이 보이는 정답 기록(확신 무관) / prev_vis8 = 그중 p0 ≥ .8;
  silent = EA2·TNA8·B5 모두 미발동(DS7 정의);  ancH = DS8 조상(v15R 사슬 ≤ 6단계, 누적 확신 ≥ .3) 이 고EC;  anc_steps.
평가: 예측 = 14회차 EC(submission_23), 짝 = v15r15_eval, 사슬 중간은 같은 ID 앞 번호 평가 기록만(규정), EA2/TNA8 = KF1E 와 같은 평가 앵커 집합.
감사 2: P2LOO·EL1 에서 DS7·DS8 은 '시험 기록만' 숨겼음 — 이웃(같은 농장 ±1 번호)까지 숨기면 상황 지표가 얼마나 바뀌나(EL1 은 원래 ±1 비움 배치).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DS8SRC = open(os.path.join(HERE, "ds8_ancestor_anchored_hmm_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ds8_ancestor_anchored_hmm_v1.py")
exec(DS8SRC[:DS8SRC.index("res, blocks, fl = [], {}, []")])   # zflag, silent, ancestor, PMT, PEVT, TNA8, V7P, DM, idx ...
__file__ = os.path.join(HERE, "vc1_validator_geometry_audit_v1.py")
def situ(v, r, g, picks, hs, vis, key):
    pv = picks.get(r, (None, 0)); vis_prev = isinstance(pv[0], str) and TRAIN[idx[pv[0]]] and pv[0] not in hs and np.isfinite(DM[idx[pv[0]]])
    an = ancestor(r, picks, hs)
    return dict(val=v, rec=r, pass2=NUM[idx[r]] >= 179, predH=g.sg.mean() >= 1.0, trueH=(DM[idx[r]] >= 1.0) if np.isfinite(DM[idx[r]]) else np.nan,
                prev_vis=bool(vis_prev), prev_vis8=bool(vis_prev and pv[1] >= .8), silent=silent(r, g, hs, vis, key),
                ancH=bool(an is not None and DM[idx[an[0]]] >= 1.0), anc_steps=(len(an[1]) + 1) if an is not None else np.nan)
rows = []
def nbr(te): return {rid(i) for i in range(N) for t in te if FARM[i] == FARM[idx[t]] and abs(int(NUM[i]) - int(NUM[idx[t]])) <= 1}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.seed == 47)].copy(); O["rec"] = O.row_id.str[:7]
for v in ("DIAG10", "P2LOO", "EL1"):
    G = O[O.validator == v].sort_values(["rec", "hour"]).reset_index(drop=True); G["cm"] = G.groupby("rec").sg.transform(lambda s: s.expanding().mean())
    for k, Gk in G.groupby("fold"):
        te = set(Gk.rec)
        for tag, hs in ((v, te), (v + "+nbr", te | nbr(te))):
            if tag == "DIAG10+nbr": continue
            vis = TRAIN & ~np.array([rid(i) in hs for i in range(N)])
            for r in te: rows.append(situ(tag, r, G[G.rec == r], PMT, hs, vis, (tag, k)))
    print("done", v, flush=True)
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) == "7")].copy().rename(columns={"SG2": "sg"})
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hidEV = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
R = R.sort_values(["rec", "hour"]).reset_index(drop=True); R["cm"] = R.groupby("rec").sg.transform(lambda s: s.expanding().mean())
for k, Gk in R.groupby("fold"):
    te = set(Gk.rec); hs = hidEV[int(k)] | te; vis = TRAIN & ~np.array([rid(i) in hs for i in range(N)])
    for r in te: rows.append(situ("EV", r, R[R.rec == r], PEVT, hs, vis, ("EV", k)))
print("done EV", flush=True)
# ---- evaluation ----
P = pd.read_csv(os.path.join(env.ROOT, "제출", "14회차_2026-10-13(나)", "submission_23.csv")); P["rec"] = P.row_id.str[:7]; P["hour"] = P.row_id.str[8:].astype(int)
P = P.rename(columns={"sub_ec": "sg"}).sort_values(["rec", "hour"]).reset_index(drop=True); P["cm"] = P.groupby("rec").sg.transform(lambda s: s.expanding().mean())
E15 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_eval.csv"))
PEVAL = {}
for r, pv, p0 in zip(E15.rec, E15.prev0, E15.p0):
    if isinstance(pv, str) and not TRAIN[idx[pv]] and not (FARM[idx[pv]] == FARM[idx[r]] and NUM[idx[pv]] < NUM[idx[r]]): continue   # rule: eval hop only same ID earlier
    PEVAL[r] = (pv, p0)
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig")
V7P_save = dict(V7P); V7P.clear(); V7P.update({r: p for r, p, ph, l, d in zip(E7.row, E7.prev_h0, E7.p_h0, E7.prev_is_labelled.astype(bool), E7.dT_air) if ph >= .8 and l and abs(d) <= 1.5})
T8_save = dict(TNA8); TNA8.clear()
hp2 = _ns["hp2"]; v15_new = None
for r, pv, p0, lb in zip(E15.rec, E15.prev0, E15.p0, E15.lab0):
    if isinstance(pv, str) and bool(lb) and r not in V7P and (p0 >= .8 or hp2(idx[r], TRAIN.copy()) == pv): TNA8[r] = pv
vis = TRAIN.copy(); hs = {rid(i) for i in range(N) if not TRAIN[i]}
for r, g in P.groupby("rec"): rows.append(situ("EVAL", r, g, PEVAL, hs, vis, ("EVAL", 0)))
V7P.clear(); V7P.update(V7P_save); TNA8.clear(); TNA8.update(T8_save)
S = pd.DataFrame(rows); pd.set_option("display.width", 220)
order = ["EVAL", "DIAG10", "P2LOO", "P2LOO+nbr", "EL1", "EL1+nbr", "EV"]
T = S.groupby("val").agg(n=("rec", "size"), pass2=("pass2", "mean"), predH=("predH", "mean"), prev_vis=("prev_vis", "mean"), prev_vis8=("prev_vis8", "mean"),
                         silent=("silent", "mean"), ancH=("ancH", "mean"), anc_steps=("anc_steps", "mean")).reindex(order)
print("\n[record situation shares]"); print(T.round(3).to_string())
for v in order:
    s = S[(S.val == v) & S.predH]
    print(f"{v:10s} predicted-high {len(s):3d}: silent {s.silent.mean():.2f} | ancH {s.ancH.mean():.2f} | prev_vis8 {s.prev_vis8.mean():.2f} | truly high {pd.to_numeric(s.trueH, errors='coerce').mean():.2f}")
print("\nEVAL predicted-high records:"); print(S[(S.val == "EVAL") & S.predH][["rec", "silent", "prev_vis", "prev_vis8", "ancH", "anc_steps"]].to_string(index=False))
S.to_csv(os.path.join(HERE, "vc1_situations_v1.csv"), index=False)
