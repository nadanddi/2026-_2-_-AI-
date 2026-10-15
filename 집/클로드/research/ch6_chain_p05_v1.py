# -*- coding: utf-8 -*-
"""CH6 (2026-10-16 집 클로드; HYB1b(7.007) 분해: 사슬 넓힘 이득은 확신 .5~.8·dT 작은 연결, 손해는 확신 < .5 -> '사슬 문턱 p >= .5' 가설).  사전 고정(커밋 후 실행).  온도.
HYB1b 는 CV8·N·CK2 에서 본 결과 -> 판정은 그 분해에 안 쓴 EXT10·EXT12·DIAG10 (TV3 틀, FL5 거울 표; LCG2 에서 본 배치이나 p .5 경계는 거기서 정하지 않음).
안 CH5: 19회차 사슬 M1(.9 exp(-h/8)) 그대로, 사슬 대상 문턱만 p >= .8 -> p >= .5 (dT <= 1.5, 같은 농장 앞 날짜, 정답 없는 같은 폴드 전날 — 나머지 동일).  K8 정답 앵커는 그대로(p >= .8).
판정(단일 안, alpha .025): CH5 vs M1(19회차) — 12셀 모두 개선 & 같은 날 중복 제거 P(악화) < .025 & 바뀐 기록 '도움 > 해'.  (12셀은 상관 높아 사실상 1관찰 — 주 판정은 P 와 기록 수.)
서술: 추가 사슬 기록 수, 기록별 dSSE 분포, 평가 링크 기준 추가 대상 기록 수.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ch6_chain_p05_v1.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "pch5_chain_mirror_links_v1.py"), encoding="utf-8").read()
src = src.replace('__file__ = os.path.join(HERE, "pch5_chain_mirror_links_v1.py")', '__file__ = os.path.join(HERE, "ch6_chain_p05_v1.py")')
exec(compile(src[:src.index("src2 = open(os.path.join(HERE, \"tv3_v3_gru_on_v23_v1.py\")")], "pch5pre", "exec"))
def chains_t(LK, recs, pos, h, FOLDOF, labelled_ok, pmin):
    AF, CH = {}, {}
    for r in recs:
        if r not in LK or not isinstance(LK[r][0], str): continue
        p, q = LK[r]
        if p not in IDX: continue
        dt = abs(IN_[IDX[r], 0, 0] - IN_[IDX[p], 23, 0])
        if not np.isfinite(dt) or dt > 1.5: continue
        if labelled_ok(r, p):
            if q >= .8 and np.isfinite(ST[IDX[p], 23]): AF[r] = float(ST[IDX[p], 23])
        elif q >= pmin and p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) and p in pos and FOLDOF.get(p) == FOLDOF.get(r) and (h[pos[p]] == 23).any():
            CH[r] = p
    return AF, CH
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3ch6"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
MIR = pd.read_csv(os.path.join(env.LOCAL, "fl5_mirror_links.csv")); RES = {}
m1 = lambda ix: S1 * np.exp(-h_g[ix] / TAU)
for L in ("EXT10", "EXT12", "DIAG10"):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); M = M[np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; h_g = h; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TXo = tx3(M, key, .25, 10, .8); FO = dict(zip(key, M.fold.values)); HELD = {}
    for r, k in FO.items(): HELD.setdefault(k, set()).add(r)
    LK = {r: (p, q) for r, p, q in zip(MIR[MIR.layout == L].rec, MIR[MIR.layout == L].prev, MIR[MIR.layout == L].p)}; order = sorted(recs, key=lambda r: (r[:3], int(r[4:7])))
    P = {}; NCH = {}
    for nm, pmin in (("M1", .8), ("CH5", .5)):
        AF, CH = chains_t(LK, recs, pos, h, FO, lambda r, p: p not in HELD[FO[r]], pmin); run = make_run(AF, CH, pos, h, TXo, order); NCH[nm] = set(CH)
        P[nm] = [run(B2.w40(M, b, cx, fam), lambda ix: S1 * np.exp(-h[ix] / TAU)) for b in bs for fam in "AB"]
    RES[L] = dict(P=P, y=y, rec=key, extra=NCH["CH5"] - NCH["M1"], bk=(M.farm + "_" + (M.day // 5).astype(str)).values)
    print(L, "chains M1", len(NCH["M1"]), "| CH5", len(NCH["CH5"]), flush=True)
rng = np.random.default_rng(20270164); cells = []; recd = []
for L in RES:
    R_ = RES[L]; y = R_["y"]; ch = [rm(a - y) / rm(b - y) - 1 for a, b in zip(R_["P"]["CH5"], R_["P"]["M1"])]; cells += ch
    dd = np.mean([(a - y) ** 2 - (b - y) ** 2 for a, b in zip(R_["P"]["CH5"], R_["P"]["M1"])], 0); S = pd.Series(dd).groupby(R_["rec"]).sum(); S = S[S.abs() > 1e-9]; recd += list(S.values)
    print(f"  CH5 vs M1 | {L}: cells {' '.join('%+.3f' % (100 * x) for x in ch)} | extra chain records {len(R_['extra'])} | changed {len(S)} helped {int((S < 0).sum())} harmed {int((S > 0).sum())} | sum {S.sum():+.2f}")
per = {L: pd.Series(np.mean([(a - RES[L]["y"]) ** 2 - (b - RES[L]["y"]) ** 2 for a, b in zip(RES[L]["P"]["CH5"], RES[L]["P"]["M1"])], 0)).groupby(RES[L]["bk"]).mean() for L in RES}
U = pd.concat(per, axis=1).mean(axis=1).values; pw = float((U[rng.integers(0, len(U), (20000, len(U)))].mean(1) >= 0).mean())
hp, hm = sum(x < 0 for x in recd), sum(x > 0 for x in recd)
print(f"CH6 CH5 (chain p >= .5) vs M1: 12 cells all better {all(x < 0 for x in cells)} | dedup P(worse) {pw:.4f} | records helped {hp} harmed {hm} -> {'PASS' if all(x < 0 for x in cells) and pw < .025 and hp > hm else 'FAIL'}")
EV = pd.read_csv(os.path.join(env.LOCAL, "fl3_eval_links.csv")); same = np.array([isinstance(p, str) and p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) for r, p in zip(EV.rec, EV.prev)])
base = same & ~EV.prev_is_labelled & (EV.dT_air.abs() <= 1.5)
print(f"  eval (links only): chain records M1 {int((base & (EV.p >= .8)).sum())} -> CH5 {int((base & (EV.p >= .5)).sum())}")
