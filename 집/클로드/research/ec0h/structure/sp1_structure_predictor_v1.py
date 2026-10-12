# -*- coding: utf-8 -*-
"""SP1 (2026-10-12 집 클로드; 사용자: '구조 기반 EC 예측기 — 앞 정보 없는 애들은 빼고, 평가 구조와 비슷한 검증으로 학습·검증').  사전 고정 — 실행 전 커밋.
검증 세계(평가 모사): vf3b 의 world_cv — 대상의 10번호 덩어리 + 앞뒤 1번호 지움 + 다른 ID 거울 덩어리(−2/+2) 숨김, 같은 ID 앞번호 기록은 보임.
구조 예측(행 h 는 대상 0..h 입력만):
  1) 2단계 모델(날짜 → 동, vf3b 그대로, 진실 확실 연결 190 로 학습, 덩어리 밖 학습 = OOF)로 보이는 전날 후보 j 의 확률 P_j(h; h 는 0/6/12/23 중 h 이하 최대).
  2) 정답 있는 후보들: P_lab = Σ P_j, L = Σ P_j·EC23_j / P_lab.
  3) q_h = sg_h + w_b · P_lab · (L − cm_h), cm_h = sg 의 0..h 누적평균(EA2 형), w_b = 시각 블록(0-5/6-11/12-22/23)별 다른 덩어리 최소제곱, [0, 1.2].
대상(앞 정보 있음, 평가에서도 같은 방식으로 판정 가능): DM11 달력에서 대상 날짜 − 1 에 그 세계에서 보이는 정답 기록이 1개 이상.
비교: BASE = DIAG10 CUR sg(시드별), HARD = v7m 1등이 정답 있는 같은 ID·p ≥ .8·보임이면 EA2 앵커(w 1), 아니면 BASE.
판정: 대상 기록에서 SP 가 HARD 보다 3시드 모두 좋음 AND 덩어리 부트스트랩 P(worse) < .025 AND 최대 기록 몫 ≤ .5.  BASE 대비·1/2차 따로 보고.
"""
import sys, os, io
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
src = io.open(os.path.join(H_, "ec0h", "verify3", "vf3b_date_house_v2.py"), encoding="utf-8").read()
src = src[:src.index("rows = []\nfor h in HS:")]
g = {"__name__": "vf3b", "__file__": os.path.join(H_, "ec0h", "verify3", "vf3b_date_house_v2.py")}; exec(compile(src, "vf3b", "exec"), g)
np, pd = g["np"], g["pd"]; world_cv, build, posterior, fit_models, REC, rid, idx = g["world_cv"], g["build"], g["posterior"], g["fit_models"], g["REC"], g["rid"], g["idx"]
TRAIN, EC, FARM, NUM, N = g["TRAIN"], g["EC"], g["FARM"], g["NUM"], g["N"]; HS = g["HS"]
OUT = os.path.dirname(os.path.abspath(__file__))
CAL = pd.read_csv(os.path.join(H_, "ec0h", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
Qs = {sd: G.groupby(["rec", "hour"]).sg.mean().unstack() for sd, G in O.groupby("seed")}; Y = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
# weights per (block, h) fitted on REC outside the block
blocks = sorted({world_cv(i)[2] for i in np.where(TRAIN)[0]})
W = {}
for blk in blocks:
    tr = [r for r in REC if r["blk"] != blk]
    for h in HS: W[(blk, h)] = fit_models(tr, h)
print("weights fitted for", len(blocks), "blocks", flush=True)
hmap = lambda h: max(x for x in HS if x <= h)
rows = []
for t in np.where(TRAIN)[0]:
    r = rid(t)
    if r not in Y.index or not np.isfinite(DAY.get(r, np.nan)): continue
    vis, lab, blk = world_cv(t)
    d0 = DAY[r]; prevlab = [j for j in np.where(vis & lab)[0] if DAY.get(rid(j)) == d0 - 1]
    if not prevlab: continue
    dates, _ = build(t, vis, lab); rec = dict(rec=r, blk=blk)
    for h in HS:
        post, _, _ = posterior(dates, *W[(blk, h)], h)
        pl = sum(p for j, p in post.items() if lab[j]); L = sum(p * EC[j, 23] for j, p in post.items() if lab[j]) / pl if pl > 0 else np.nan
        rec[f"Pl{h}"], rec[f"L{h}"] = pl, L
    hard = None
    if r in L7.index:
        pr = L7.loc[r]; j = idx.get(pr.prev)
        if j is not None and not pr.xf and pr.p >= .8 and lab[j] and vis[j]: hard = float(EC[j, 23])
    rec["hard"] = hard if hard is not None else np.nan
    rows.append(rec)
D = pd.DataFrame(rows).set_index("rec"); D.to_csv(os.path.join(OUT, "sp1_posteriors_v1.csv"))
print("eligible records (front info):", len(D), "| pass-2:", int((D.index.str[4:7].astype(int) >= 179).sum()), "| HARD anchors:", int(D.hard.notna().sum()), flush=True)
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e)))); rng = np.random.default_rng(20261247)
recs = list(D.index); out = {}
for sd, Q in Qs.items():
    Q = Q.loc[recs]; cm = Q.cumsum(1) / np.arange(1, 25); Yr = Y.loc[recs]
    Xs = np.zeros((len(recs), 24))
    for h in range(24):
        hh = hmap(h); Xs[:, h] = D[f"Pl{hh}"].fillna(0).values * (D[f"L{hh}"].fillna(0).values - cm[h].values) * (D[f"Pl{hh}"].fillna(0).values > 0)
    SP = Q.copy(); bl = lambda h: 0 if h < 6 else (1 if h < 12 else (2 if h < 23 else 3))
    for blk in D.blk.unique():
        tr = (D.blk != blk).values; te = ~tr
        for b in range(4):
            hs = [h for h in range(24) if bl(h) == b]
            x = Xs[tr][:, hs].ravel(); y = (Yr.values[tr][:, hs] - Q.values[tr][:, hs]).ravel(); ok = np.isfinite(y)
            w = float(np.clip((x[ok] * y[ok]).sum() / (x[ok] ** 2).sum(), 0, 1.2)) if (x[ok] ** 2).sum() > 0 else 0.0
            SP.values[np.ix_(np.where(te)[0], hs)] = Q.values[np.ix_(np.where(te)[0], hs)] + w * Xs[np.ix_(np.where(te)[0], hs)]
    HD = Q.copy(); hm = D.hard.notna().values
    HD.values[hm] = Q.values[hm] + (D.hard.values[hm][:, None] - cm.values[hm])
    out[sd] = dict(Q=Q, SP=SP, HD=HD, Y=Yr)
for comp in ("HD", "Q"):
    for part, cond in (("all", lambda s: True), ("pass-1", lambda s: int(s[4:7]) < 179), ("pass-2", lambda s: int(s[4:7]) >= 179)):
        rr = [r for r in recs if cond(r)]
        if not rr: continue
        cells = [rm((o["SP"].loc[rr] - o["Y"].loc[rr]).values) / rm((o[comp].loc[rr] - o["Y"].loc[rr]).values) - 1 for o in out.values()]
        per = pd.Series(np.mean([(((o["SP"].loc[rr] - o["Y"].loc[rr]) ** 2).sum(1) - ((o[comp].loc[rr] - o["Y"].loc[rr]) ** 2).sum(1)).values for o in out.values()], 0), index=rr)
        cl = per.groupby(D.blk.reindex(rr).values).sum(); ii = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[ii].sum(1) >= 0).mean())
        share = float(per.min() / per.sum()) if per.sum() < 0 else float("nan")
        base_rm = np.mean([rm((o[comp].loc[rr] - o["Y"].loc[rr]).values) for o in out.values()])
        print(f"SP vs {'HARD' if comp == 'HD' else 'BASE'} {part:6s}: n {len(rr)} | {comp} RMSE {base_rm:.4f} | cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | improved {int((per<0).sum())}/{len(per)} | block P {pw:.4f} | top share {share:.2f}")
        if comp == "HD" and part == "all": print("   VERDICT SP1:", "PASS" if all(x < 0 for x in cells) and pw < .025 and share <= .5 else "FAIL")
