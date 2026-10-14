# -*- coding: utf-8 -*-
"""LA1 (2026-10-15 집 클로드; 사용자: 'lk5 짝 넣어서 정식 시험 진행해봐').  사전 고정(커밋 후 실행).  온도 + EC (판정 따로).
지금 앵커 짝: 온도 LV = 링커 v7 확신(cv_all400, p≥.8·정답 전날·|dT|≤1.5) + TNA8(v15R 거울), EC = v7 확신(EA 식, 6.574).
LK5(917a34b7): LK4 후보 순위(h 0) + 짝 신뢰 모형 → 앵커 짝 정확도 .872(47 중 41) vs v15R .837(43 중 36), EV cert 대상 기준.
여기선 LK5 짝 표를 400 대상 전부(cert 아닌 대상 포함 — 실제 배치)에 대해 거울 기하(mirror_lab, 지금 LV 와 같은 기하)로 만들고 앵커만 바꿔 점수 비교.
LK5 짝 표 만들기(규칙 미리 고정):
  후보·특징 = LK4 cand_rows(h 0 특징만 사용), 대상 = 400 전부.  LK4 순위 모형: cert 대상 후보 행으로 적합 — cert 대상 점수는 대상 5겹 교차 적합, 나머지 대상은 cert 전체 적합.
  신뢰 모형: LK5 와 같은 16특징, cert 대상 OOF 짝으로 적합(5겹 교차 적합으로 cert 대상 신뢰도 산출, 나머지는 전체 적합).
  앵커 짝 = 1등이 보이는 정답 기록 & 신뢰 ≥ 문턱 t,  t = cert 대상 OOF 에서 '보이는 짝 정확도 ≥ .90' 이 되는 가장 낮은 값(LK5 규칙).
  anchor_st23 = 짝 23시 배지온 정답, dT_air = 대상 0시 − 짝 23시 실내온(v7 표와 같은 정의), p = 1(앵커 함수 문턱 통과용), 앵커 식·감쇠·게이트는 그대로.
[온도] 판정(TV3 틀, 단일 안 α .025): v23 의 LV 를 LK5 짝으로 바꾼 판이 DIAG10·EL1·F·G·H·I·K 모든 셀 개선 & 합친(DIAG10+F~K, 셀 1) 5일 블록 P(악화) < .025
       & EXT10·EXT12 안전(셀 평균 악화 & '나음' 비율 < .025 이면 FAIL).
[EC]  ec_rv1 CUR(DIAG10·EL1·P2LOO, 시드 3) sg 위 EA 식 앵커: CUR 짝 vs LK5 짝(폴드에서 숨긴 기록이 짝이면 그 앵커는 두 안 모두 안 씀).
       판정(EC 분야 배정 규칙): P2LOO·EL1 모든 시드 2차 dSSE(LK5 − CUR) < 0 & (P2LOO 블록 P(악화) < .025 또는 나빠진 2차 기록-칸 0); DIAG10 손해 감시.
서술: 앵커 수(CUR / LK5 / 공통), 짝이 바뀐 대상, cert 진짜 전날과의 일치율, 장부 기록 제외 dSSE.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u la1_lk5_anchor_links_formal_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "lk4_candidate_ranker_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "lk4_candidate_ranker_v1.py")
exec(src[:src.index("MT = build(")])
__file__ = os.path.join(HERE, "la1_lk5_anchor_links_formal_v1.py")
from feat4 import ST, EC as ECA  # noqa: E402
rng0 = np.random.default_rng(0)
PMr = PM.reset_index(); allT = []
for _, q in PMr.iterrows():
    D = cand_rows(q, mirror_lab(q.rec))
    if len(D): D["cert"] = q.rec in CERT; allT.append(D)
A = pd.concat(allT, ignore_index=True); A.loc[~A.cert, "ok"] = np.nan
print("targets", A.rec.nunique(), "(cert", int(A.groupby("rec").cert.first().sum()), ") candidate rows", len(A), flush=True)
F = [f"fp0" if f == "fp" else ("fp_rank0" if f == "fp_rank" else ("fp_ratio0" if f == "fp_ratio" else f)) for f in FEATS]
recs = np.array(sorted(A[A.cert].rec.unique())); fold = {r: k for k, r in zip(rng0.permutation(len(recs)) % 5, recs)}
def fitscore(tr, te):
    sc = StandardScaler().fit(tr[F]); LR = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(tr[F]), tr.ok.astype(int)); return LR.decision_function(sc.transform(te[F]))
A["s"] = np.nan; C_ = A[A.cert]
for k in range(5):
    m = A.cert & (A.rec.map(fold) == k); A.loc[m, "s"] = fitscore(C_[C_.rec.map(fold) != k], A[m])
A.loc[~A.cert, "s"] = fitscore(C_, A[~A.cert])
pn = PM.p_null.to_dict(); P = []
for r, d in A.groupby("rec"):
    d = d.sort_values("s", ascending=False); t = d.iloc[0]; s2 = d.s.iloc[1] if len(d) > 1 else t.s - 5; f2 = d.fp0.iloc[1] if len(d) > 1 else np.nan
    P.append(dict(rec=r, cand=t.cand, ok=t.ok, cert=t.cert, s1=t.s, gap=t.s - s2, q=float(1 / np.exp(d.s - t.s).sum()), n_cand=t.n_cand, date_ok=t.date_ok, date_bad=t.date_bad, date_unk=t.date_unk,
                  fp=t.fp0, runner=(f2 / max(t.fp0, 1e-3)) if np.isfinite(f2) else 3.0, dTin=t.dTin, dTout=t.dTout, cross=t.cross, v15p=t.v15p, v15rank=t.v15rank, vis=t.cand_vis, p_null=float(np.nan_to_num(pn.get(r, 0.0)))))
P = pd.DataFrame(P); TF = ["s1", "gap", "q", "n_cand", "date_ok", "date_bad", "date_unk", "fp", "runner", "dTin", "dTout", "cross", "v15p", "v15rank", "vis", "p_null"]
P["t"] = np.nan; Pc = P[P.cert]
def trust(tr, te):
    sc = StandardScaler().fit(tr[TF]); LR = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(tr[TF]), tr.ok.astype(int)); return LR.predict_proba(sc.transform(te[TF]))[:, 1]
for k in range(5):
    m = P.cert & (P.rec.map(fold) == k); P.loc[m, "t"] = trust(Pc[Pc.rec.map(fold) != k], P[m])
P.loc[~P.cert, "t"] = trust(Pc, P[~P.cert])
vm = P[P.cert & (P.vis == 1)].sort_values("t", ascending=False); vm["cacc"] = vm.ok.expanding().mean(); thr = float(vm[vm.cacc >= .90].t.min())
SEL = P[(P.vis == 1) & (P.t >= thr)]
print(f"trust threshold {thr:.3f} | LK5 anchors {len(SEL)} (cert {int(SEL.cert.sum())}, cert acc {SEL[SEL.cert].ok.mean():.3f}; non-cert {int((~SEL.cert).sum())})", flush=True)
L5 = pd.DataFrame(dict(rec=SEL.rec.values, prev=SEL.cand.values, p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in SEL.cand], dT_air=[float(TIN.at[r, 0] - TIN.at[p, 23]) for r, p in zip(SEL.rec, SEL.cand)], steps_to_anchor=1))
L5 = L5[np.isfinite(L5.anchor_st23)]
P.to_csv(os.path.join(HERE, "la1_lk5_picks_v1.csv"), index=False)
# ---------------- temperature ----------------
src = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3la1"}
exec(src[:src.index("JUDGE = [")], g)
T_, B_, C2_, anch_, LV, D1, CT, rmf = g["T"], g["B"], g["C2"], g["anch"], g["LV"], g["D1"], g["CT"], g["rm"]
cur = dict(zip(LV.rec, LV.prev)); new = dict(zip(L5.rec, L5.prev))
print(f"[TEMP] anchors CUR {len(cur)} | LK5 {len(new)} | same pick {sum(1 for r in cur if new.get(r) == cur[r])} | changed {sum(1 for r in cur if r in new and new[r] != cur[r])} | only CUR {len(set(cur) - set(new))} | only LK5 {len(set(new) - set(cur))}")
print(f"        cert agreement: CUR {np.mean([CERT.get(r) == p for r, p in cur.items() if r in CERT]):.3f} (n {sum(r in CERT for r in cur)}) | LK5 {np.mean([CERT.get(r) == p for r, p in new.items() if r in CERT]):.3f} (n {sum(r in CERT for r in new)})", flush=True)
rng = np.random.default_rng(20261226); JUDGE = ["DIAG10", "EL1", "F", "G", "H", "I", "K"]; ok_t = True; diffs = []; safety = {}
for L in JUDGE + ["EXT10", "EXT12"]:
    d, pre, bs, cx = T_.LAY[L]; M = T_.layout_frame(L, CT); y = M.sub_temp.values; key = (M.farm + "_" + (M.day // 5).astype(str)).values; base, nw = [], []
    for bi, b in enumerate(bs):
        for fam in "AB":
            q_ = C2_.corrected(M, B_.w40(M, b, cx, fam), D1, bi, fam, 7); base.append(anch_(M, q_, LV)); nw.append(anch_(M, q_, L5))
    ch = [rmf(n - y) / rmf(b_ - y) - 1 for n, b_ in zip(nw, base)]; p2 = M.day.values >= 179
    print(f"  {L:6s} all {' '.join('%+.2f' % (100 * x) for x in ch)} | p2 {' '.join('%+.2f' % (100 * (rmf((n - y)[p2]) / rmf((b_ - y)[p2]) - 1)) for n, b_ in zip(nw, base)) if p2.any() else '-'}", flush=True)
    if L in JUDGE:
        ok_t &= all(x < 0 for x in ch)
        if L != "EL1": diffs.append(pd.Series((nw[0] - y) ** 2 - (base[0] - y) ** 2).groupby(key).mean())
    else:
        pw = [float((pd.Series((n - y) ** 2 - (b_ - y) ** 2).groupby(key).mean().values[rng.integers(0, len(np.unique(key)), (20000, len(np.unique(key))))].mean(1) < 0).mean()) for n, b_ in zip(nw, base)]
        safety[L] = not (np.mean(ch) > 0 and min(pw) < .025); ok_t &= safety[L]
S = pd.concat(diffs, axis=1).mean(axis=1).values; Pt = float((S[rng.integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
print(f"[TEMP] pooled P(worse) {Pt:.4f} | safety {safety} -> LA1-TEMP: {'ADOPT' if ok_t and Pt < .025 else 'FAIL'}", flush=True)
# ---------------- EC ----------------
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
L7 = g["L7"]; v7set = g["v7set"]
CURE = {r: p for r, p, q, l, dT in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if isinstance(p, str) and q >= .8 and l and abs(dT) <= 1.5 and r in v7set}
NEWE = {r: p for r, p in new.items() if np.isfinite(ECA[idx[p], 23])}
print(f"[EC] anchors CUR {len(CURE)} | LK5 {len(NEWE)} | changed {sum(1 for r in CURE if r in NEWE and NEWE[r] != CURE[r])}", flush=True)
def ea(p, prev):
    if prev is None or not np.isfinite(ECA[idx[prev], 23]): return p
    pm = np.cumsum(p) / np.arange(1, len(p) + 1); return p + (ECA[idx[prev], 23] - pm)
LEDGER = {"F13_231", "F13_233", "F47_216", "F47_229", "F47_241", "F13_217", "F13_243", "F47_231"}
rows, blocks, recd = [], {}, []
for (v, k, sd), G in O[O.validator.isin(["DIAG10", "EL1", "P2LOO"])].groupby(["validator", "fold", "seed"]):
    hid = set(G.rec)
    for r, gg in G.groupby("rec"):
        gg = gg.sort_values("hour"); y = gg.sub_ec.values; p = gg.sg.values
        a = CURE.get(r); a = a if a not in hid else None; b = NEWE.get(r); b = b if b not in hid else None
        rows.append(pd.DataFrame(dict(val=v, seed=sd, rec=r, day=gg.day.values, y=y, cur=ea(p, a), new=ea(p, b))))
B = pd.concat(rows, ignore_index=True); B["d"] = (B.new - B.y) ** 2 - (B.cur - B.y) ** 2; B["p2"] = B.day >= 179
res = []
for (v, sd), gg in B.groupby(["val", "seed"]):
    m = gg.p2.values; res.append(dict(val=v, seed=sd, dSSE_p2=float(gg.d[m].sum()), dSSE_all=float(gg.d.sum()), rel_p2=100 * (rmf((gg.new - gg.y)[m]) / rmf((gg.cur - gg.y)[m]) - 1) if m.any() else np.nan,
                                     rel_all=100 * (rmf(gg.new - gg.y) / rmf(gg.cur - gg.y) - 1), X0_p2=float(gg.d[m & ~gg.rec.isin(LEDGER).values].sum())))
R = pd.DataFrame(res); print(R.round(3).to_string(index=False))
def boot(v, p2only):
    gg = B[(B.val == v) & (B.p2 if p2only else True)]; s = gg.groupby(["seed", gg.rec.str[:3] + "_" + (gg.day // 5).astype(str)]).d.mean().groupby(level=1).mean().values
    m = s[rng.integers(0, len(s), (20000, len(s)))].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
main = R[R.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); pP, _ = boot("P2LOO", True)
RD = B[B.val.isin(["P2LOO", "EL1"]) & B.p2].groupby(["val", "seed", "rec"]).d.sum(); nh = int((RD > 1e-6).sum())
dg = R[R.val == "DIAG10"]; _, pb = boot("DIAG10", False); c3 = not (dg.dSSE_all.mean() > 0 and pb < .025)
print(f"[EC] (1) main cells better {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P(worse) {pP:.4f} | harmed record-cells {nh} | (3) DIAG10 mean dSSE {dg.dSSE_all.mean():+.3f} share(better) {pb:.3f}")
print("[EC] changed records dSSE (seed mean, all validators):", B.groupby("rec").d.sum().div(3).loc[lambda s: s.abs() > 1e-6].round(3).sort_values().to_dict())
print("LA1-EC:", "ADOPT" if (c1 and (pP < .025 or nh == 0) and c3) else "FAIL")
