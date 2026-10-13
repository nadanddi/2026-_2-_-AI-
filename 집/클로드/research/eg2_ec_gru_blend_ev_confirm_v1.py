# -*- coding: utf-8 -*-
"""EG2 (2026-10-14 집 클로드; 사용자: '섞는 비율 올려서 새 배치로 확인').  사전 고정(커밋 후 실행).  EC.
EG1(6.830): 하루 시퀀스 2층 GRU 10% — 9칸 모두 개선, P2LOO 효과 작아 FAIL.  여기선 비율을 올리고 EG1 에서 안 쓴 배치로 확인.
새 배치 = EV(8폴드, 평가 배치 흉내 — 지운 날·이웃 숨김; 기준 SG2, 시드 7/101/2024) — EG1 의 GRU·비율 결정에 쓰지 않음.  평가가 전부 2차라 판정은 EV 2차 행.
GRU: EG1 과 같은 구조·학습(자기 기록 0..h 입력, 2층 128, 25 에폭, 시드 1·2), 폴드마다 정답 보임 = 학습 기록 & 그 폴드 val∪del 아님.
안(2개, Bonferroni α .0125):  B20 = .8·SG2 + .2·GRU,  B30 = .7·SG2 + .3·GRU.
판정: EV 2차 행에서 3시드 모두 ΔSSE < 0  AND  시드 평균 농장×5일 블록 부트스트랩 P(악화) < .0125;  손해 감시: EV 전체 행 시드 평균 악화 AND '후보 나음' 비율 < .0125 이면 FAIL.
서술(이미 본 배치): 같은 비율의 P2LOO·EL1·DIAG10 (EG1 GRU 예측 재사용).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "eg1_ec_dayseq_gru_member_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "eg1_ec_dayseq_gru_member_v1.py")
exec(src[:src.index("O = pd.read_csv(")])   # CUBE, fit_predict, LABREC, EC, DM, idx, rm
__file__ = os.path.join(HERE, "eg2_ec_gru_blend_ev_confirm_v1.py")
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")].copy()
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hidEV = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
GE = {}
for k, Gk in R[R.seed.astype(str) == "7"].groupby("fold"):
    te = sorted(set(Gk.rec)); hide = hidEV[int(k)] | set(te); tr = [r for r in LABREC if r not in hide]
    for r, p in zip(te, fit_predict(tr, te)): GE[r] = p
    print("EV fold", k, "done", flush=True)
res, blocks = [], {}
def score(v, sd, G, base, gr, w, rowmask_name):
    y = G.y.values; new = (1 - w) * base + w * gr; p2 = G.day.values >= 179; hi = np.array([DM[idx[r]] >= 1.0 for r in G.rec])
    f = lambda m: 100 * (rm((new - y)[m]) / rm((base - y)[m]) - 1) if m.any() else np.nan
    d = (new - y) ** 2 - (base - y) ** 2; arm = f"B{int(w * 100)}"
    res.append(dict(val=v, seed=sd, arm=arm, dSSE_all=float(d.sum()), dSSE_p2=float(d[p2].sum()), all=f(np.ones(len(y), bool)), p2=f(p2), p2_high=f(p2 & hi), p2_norm=f(p2 & ~hi)))
    bk = (G.farm + "_" + (G.day // 5).astype(str)).values
    blocks.setdefault((v, arm, "all"), []).append(pd.Series(d).groupby(bk).mean())
    blocks.setdefault((v, arm, "p2"), []).append(pd.Series(d[p2]).groupby(bk[p2]).mean())
for sd, G in R.groupby(R.seed.astype(str)):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); gr = np.array([GE[r][h] for r, h in zip(G.rec, G.hour)])
    for w in (.2, .3): score("EV", sd, G, G.SG2.values, gr, w, "")
GP = pd.read_csv(os.path.join(HERE, "eg1_gru_preds_v1.csv")); GRU1 = {(v, r): row[[f"h{h}" for h in range(24)]].values.astype(float) for v, r, (_, row) in zip(GP.val, GP.rec, GP.iterrows())}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), G in O[O.validator.isin(["DIAG10", "P2LOO", "EL1"])].groupby(["validator", "seed"]):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True).rename(columns={"sub_ec": "y"}); gr = np.array([GRU1[(v, r)][h] for r, h in zip(G.rec, G.hour)])
    for w in (.2, .3): score(v + "(seen)", str(sd), G, G.sg.values, gr, w, "")
S = pd.DataFrame(res); pd.set_option("display.width", 220); print(S.round(3).to_string(index=False))
def boot(key):
    s = pd.concat(blocks[key], axis=1).mean(axis=1).values; ix = np.random.default_rng(20261203).integers(0, len(s), (20000, len(s))); m = s[ix].mean(1)
    return float((m >= 0).mean()), float((m < 0).mean())
for arm in ("B20", "B30"):
    e = S[(S.val == "EV") & (S.arm == arm)]; pw, _ = boot(("EV", arm, "p2")); _, pb = boot(("EV", arm, "all"))
    c1 = bool((e.dSSE_p2 < 0).all()) and pw < .0125; c2 = not (e.dSSE_all.mean() > 0 and pb < .0125)
    print(f"\n== EG2 {arm}: EV pass-2 cells better {int((e.dSSE_p2 < 0).sum())}/3, P(worse) {pw:.4f} -> {'ok' if c1 else 'FAIL'} | EV all-rows guard share(better) {pb:.3f} mean dSSE {e.dSSE_all.mean():+.3f} -> {'ok' if c2 else 'FAIL'}")
    print(f"EG2 {arm}:", "ADOPT" if c1 and c2 else "FAIL")
pd.DataFrame([dict(rec=r, **{f"h{h}": p[h] for h in range(24)}) for r, p in GE.items()]).to_csv(os.path.join(HERE, "eg2_gru_ev_preds_v1.csv"), index=False)
