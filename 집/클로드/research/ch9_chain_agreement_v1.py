# -*- coding: utf-8 -*-
"""CH9 (2026-10-16 집 클로드; CH8(7.010) '다른 링커 일치가 진짜 어제를 가름(AUC .853)' -> 사용자 '진행해').  사전 고정(커밋 후 실행).  온도.
안 AG4: 사슬 대상 = (CH5 와 같음: p >= .5, dT <= 1.5, 정답 없는 같은 폴드 같은 농장 앞 날짜 전날) OR (p >= .3 이고 v7·v10m·v15R·FLN 네 링커가 모두 같은 전날 선택, dT <= 1.5).
기준 = CH5(19회차 온도 + 사슬 p >= .5; 7.008).  M1 가중·K8 정답 앵커(p >= .8)·TX3 그대로.
CH8 은 EXT10·EXT12·DIAG10 에서 봄 -> 판정은 CV8·N·CK2 (FL7 거울 표) + EL1 안전.  기준(단일 안, alpha .025): 12셀 모두 개선 & 같은 날 중복 제거 P(악화) < .025 & 바뀐 기록 도움 > 해 & EL1 안전.
서술: 추가 사슬 기록 수·기록 dSSE, 평가(링크·일치) 기준 추가 대상 수.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ch9_chain_agreement_v1.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "pch5_chain_mirror_links_v1.py"), encoding="utf-8").read()
src = src.replace('__file__ = os.path.join(HERE, "pch5_chain_mirror_links_v1.py")', '__file__ = os.path.join(HERE, "ch9_chain_agreement_v1.py")')
exec(compile(src[:src.index("src2 = open(os.path.join(HERE, \"tv3_v3_gru_on_v23_v1.py\")")], "pch5pre", "exec"))
V7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7.csv"), encoding="utf-8-sig").set_index("rec").prev
V10 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "cv_all400_choice_v10m.csv"), encoding="utf-8-sig").set_index("rec").prev
V15 = pd.read_csv(os.path.join(env.LOCAL, "v15r_mirror_masked_pkg25.csv")).set_index("rec").prev0; FLNp = pd.read_csv(os.path.join(env.LOCAL, "fln1_train_links.csv")).set_index("rec").prev
nagree = lambda r, p: sum(float(T_.get(r) == p) for T_ in (V7, V10, V15, FLNp))
def plan(LK, recs, pos, h, FOLDOF, lab_ok, ag):
    AF, CH = {}, {}
    for r in recs:
        if r not in LK or not isinstance(LK[r][0], str): continue
        p, q = LK[r]
        if p not in IDX: continue
        dt = abs(IN_[IDX[r], 0, 0] - IN_[IDX[p], 23, 0])
        if not np.isfinite(dt) or dt > 1.5: continue
        if lab_ok(r, p):
            if q >= .8 and np.isfinite(ST[IDX[p], 23]): AF[r] = float(ST[IDX[p], 23])
        elif p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) and p in pos and FOLDOF.get(p) == FOLDOF.get(r) and (h[pos[p]] == 23).any():
            if q >= .5 or (ag and q >= .3 and nagree(r, p) == 4): CH[r] = p
    return AF, CH
MIR7 = pd.read_csv(os.path.join(env.LOCAL, "fl7_mirror_links.csv")); RES = {}
for L, ckb, folds, mk in (("CV8", C8.CKB, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, LF["N"], O1.masks), ("CK2", O1.CKB, LF["CK2"], O1.masks), ("EL1", CKR, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TXo = tx3(M, key, .25, 10, .8); TRS, VF = {}, {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs; VF[r] = k
    LK = {r: (p, q) for r, p, q in zip(MIR7[MIR7.layout == L].rec, MIR7[MIR7.layout == L].prev, MIR7[MIR7.layout == L].p)}; order = sorted(recs, key=lambda r: (r[:3], int(r[4:7])))
    P = {}; NCH = {}
    for nm, ag in (("CH5", False), ("AG4", True)):
        AF, CH = plan(LK, recs, pos, h, VF, lambda r, p: r in TRS and p in TRS[r], ag); run = make_run(AF, CH, pos, h, TXo, order); NCH[nm] = set(CH)
        P[nm] = [run(B.w40(M, "base_R_%d" % s, "codex_R", fam), lambda ix: S1 * np.exp(-h[ix] / TAU)) for s in O1.SEEDS for fam in "AB"]
    RES[L] = dict(P=P, y=y, rec=key, extra=NCH["AG4"] - NCH["CH5"], bk=(M.farm + "_" + (M.day // 5).astype(str)).values); print(L, "chains CH5", len(NCH["CH5"]), "| AG4", len(NCH["AG4"]), flush=True)
rng = np.random.default_rng(20270165); cells = []; recd = []
for L in RES:
    R_ = RES[L]; y = R_["y"]; ch = [rm(a - y) / rm(b - y) - 1 for a, b in zip(R_["P"]["AG4"], R_["P"]["CH5"])]
    dd = np.mean([(a - y) ** 2 - (b - y) ** 2 for a, b in zip(R_["P"]["AG4"], R_["P"]["CH5"])], 0); S = pd.Series(dd).groupby(R_["rec"]).sum(); S = S[S.abs() > 1e-9]
    if L != "EL1": cells += ch; recd += list(S.values)
    print(f"  AG4 vs CH5 | {L}: cells {' '.join('%+.3f' % (100 * x) for x in ch)} | extra {len(R_['extra'])} | changed {len(S)} helped {int((S < 0).sum())} harmed {int((S > 0).sum())} | sum {S.sum():+.2f}")
per = {L: pd.Series(np.mean([(a - RES[L]["y"]) ** 2 - (b - RES[L]["y"]) ** 2 for a, b in zip(RES[L]["P"]["AG4"], RES[L]["P"]["CH5"])], 0)).groupby(RES[L]["bk"]).mean() for L in ("CV8", "N", "CK2")}
U = pd.concat(per, axis=1).mean(axis=1).values; pw = float((U[rng.integers(0, len(U), (20000, len(U)))].mean(1) >= 0).mean())
R_ = RES["EL1"]; c2 = [rm(a - R_["y"]) / rm(b - R_["y"]) - 1 for a, b in zip(R_["P"]["AG4"], R_["P"]["CH5"])]; safe = not (np.mean(c2) > .001)
hp, hm = sum(x < 0 for x in recd), sum(x > 0 for x in recd)
print(f"CH9 AG4 vs CH5: 12 cells all better {all(x < 0 for x in cells)} | dedup P(worse) {pw:.4f} | records helped {hp} harmed {hm} | EL1 {100 * np.mean(c2):+.3f}% -> {'PASS' if all(x < 0 for x in cells) and pw < .025 and hp > hm and safe else 'FAIL'}")
