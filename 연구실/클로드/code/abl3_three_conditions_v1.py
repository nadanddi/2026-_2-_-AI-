# -*- coding: utf-8 -*-
"""ABL3 (2026-11-03 연구실 클로드; 사용자 '실험 돌려줘' — 코덱스 근거확인 v2 8절의 세 조건 비교).
서술용(채택 판정 아님). 실행 전 이 파일의 SHA256을 작업일지에 적어 고정. 학습·평가 예측·제출물 변경 없음.
틀은 ABL1(집/클로드/research/abl1_link_modules_off_v1.py)과 같음: 온도 TXP2 34 구조(= 35e 온도), EC ECT2 앵커(= 35e EC).

세 조건
  조건 1 '어제를 앎'(오라클) : 확실한 짝(linker_v3/truth.pkl cert, 학습 F13·F47 190쌍)의 진짜 어제를 씀.
        온도: 진짜 어제가 그 폴드에서 정답 있는 기록이면 앵커(어제 23시 실측), 가려진 기록이면 사슬(어제 최종 23시 예측). 다른 ID여도 씀(현장에서는 동이 알려짐).
        EC  : 진짜 어제가 정답 있는 기록이면 앵커(강도 .8·q, q = 1). 가려진 어제면 앵커 없음(EC 는 사슬 없음).
        진짜 어제를 모르는 기록(짝 밖)은 최종 구성의 링커 결과를 그대로 둠. 진짜 어제를 아는데 쓸 수 없으면 아무것도 안 붙임.
        O1 = 실내 자정 변화 |dT| <= 1.5 조건 유지, O2 = 이 조건 없음.
  조건 2 '링커로 찾은 어제' : 최종 구성(온도 A3, EC E2).
  조건 3 '어제 정보 없음'   : 온도 A1(NX+TX3), EC E0(pre+HR5).
보고: 검증기·조건별 RMSE(온도 셀 평균, EC 시드 평균), 범위 = 전체·2차 행·짝 기록(진짜 어제를 아는 기록)·짝 기록 2차.
      바뀐 기록 수(최종 계획이 이미 진짜 어제를 쓴 기록 / 다른 어제 / 안 씀), 블록 부트스트랩(5일 블록) P(앞 조건이 뒤 조건보다 나쁘거나 같음).
해석 주의(미리 적음): 짝은 대상 기록의 0·1시 정답이 앞 기록 23시와 이어지는 것으로 골랐으므로, 짝 기록에서 조건 1은 구조상 낙관적(진단용 오라클).
      1 대 2 = 링커 오류의 손해(상한 쪽 추정), 1 대 3 = 어제 정보의 가치(상한 쪽). 계절 지표는 모든 조건에 남음. 링커 가중은 전체 학습 자료로 맞춘 값.
Run:  cd 연구실/클로드/code && PYTHONPATH="" python -u abl3_three_conditions_v1.py
"""
import os, sys
RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "집", "클로드", "research"))
OUTD = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "local", "abl3"))
_ME = os.path.abspath(__file__)
sys.path.insert(0, RES); os.chdir(RES)
__file__ = os.path.join(RES, "abl3_three_conditions_v1.py")
import env  # noqa: F401
# ---------------- 온도 ----------------
_src = open(os.path.join(RES, "txp2_tx3_pass2_v1.py"), encoding="utf-8").read()
_src = _src[:_src.index("# ---- TXP2:")]
exec(compile(_src, "txp2pre", "exec"))
__file__ = os.path.join(RES, "abl3_three_conditions_v1.py")
rmse = lambda e: float(np.sqrt(np.nanmean(np.square(e))))

def topo(recs, CH):
    seen, out = set(), []
    def go(r, stack=()):
        if r in seen or r in stack: return
        if r in CH: go(CH[r], stack + (r,))
        seen.add(r); out.append(r)
    for r in sorted(recs, key=lambda r: (r[:3], int(r[4:7]))): go(r)
    return out

def oracle_temp(F, gate_dt):
    M = F["M"]; FO = dict(zip(F["key"], M.fold.values)); HELD = {}
    for r, k in FO.items(): HELD.setdefault(k, set()).add(r)
    A, C = dict(F["A_tag"]), dict(F["CHfull"]); stat = {"same": 0, "diff": 0, "none_before": 0, "unusable": 0}
    for r in F["pos"]:
        if r not in TRUEP: continue
        p = TRUEP[r]; A.pop(r, None); C.pop(r, None)
        ok_dt = True
        if p in IDX:
            dt = abs(IN_[IDX[r], 0, 0] - IN_[IDX[p], 23, 0]); ok_dt = (not gate_dt) or (np.isfinite(dt) and dt <= 1.5)
        if p in IDX and ok_dt and p not in HELD[FO[r]] and np.isfinite(ST[IDX[p], 23]): A[r] = float(ST[IDX[p], 23])
        elif ok_dt and p in HELD[FO[r]] and p in F["pos"] and (F["h"][F["pos"][p]] == 23).any(): C[r] = p
        else: stat["unusable"] += 1
    return A, C, topo(F["pos"].keys(), C), stat

def plan_vs_truth(F):
    # 최종 계획이 짝 기록에 쓴 어제가 진짜 어제와 같은지: 앵커 값은 사전에 어제 이름이 없어 FL5/FL8 표로 복원
    MIRL = {r: p for r, p in zip(MIR5[MIR5.layout == F["L"]].rec, MIR5[MIR5.layout == F["L"]].prev)}
    F8L = {r: p for r, p in zip(F8[F8.layout == F["L"]].rec, F8[F8.layout == F["L"]].prev)}
    c = {"anchor_same": 0, "anchor_diff": 0, "chain_same": 0, "chain_diff": 0, "none": 0}
    for r in F["pos"]:
        if r not in TRUEP: continue
        t = TRUEP[r]
        if r in F["A_tag"]:
            p = MIRL.get(r) if r in F["AF"] else F8L.get(r); c["anchor_same" if p == t else "anchor_diff"] += 1
        elif r in F["CHfull"]: c["chain_same" if F["CHfull"][r] == t else "chain_diff"] += 1
        else: c["none"] += 1
    return c

TROWS = []; TBLK = {}
for L, F in FR.items():
    F["L"] = L
    y = F["y"]; p2 = F["M"].day.values >= 179; ok = np.isfinite(y); cert = np.isin(F["key"], list(TRUEP))
    O = {g: oracle_temp(F, g) for g in (True, False)}
    P = {a: [] for a in ("C3", "C2", "O1", "O2")}
    for b in F["bs"]:
        for fam in "AB":
            x = B2.w40(F["M"], b, F["cx"], fam)
            P["C3"].append(x + F["TXo"])
            P["C2"].append(assemble2(x, F["A_tag"], F["CHfull"], F["pos"], F["h"], F["TXo"], F["order"], .9, 8.0, {}))
            for nm, g in (("O1", True), ("O2", False)):
                A, C, od, _ = O[g]; P[nm].append(assemble2(x, A, C, F["pos"], F["h"], F["TXo"], od, .9, 8.0, {}))
    scopes = (("all", ok), ("p2", p2 & ok), ("cert", cert & ok), ("cert_p2", cert & p2 & ok))
    for a in P:
        for sc, m in scopes:
            TROWS.append(dict(target="temp", val=L, arm=a, scope=sc, n=int(m.sum()), rmse=float(np.mean([rmse((p - y)[m]) for p in P[a]]))))
    for sc, m in scopes:
        for pair in (("O1", "C2"), ("C2", "C3"), ("O1", "C3")):
            d = np.mean([(p - y) ** 2 - (q - y) ** 2 for p, q in zip(P[pair[0]], P[pair[1]])], 0)
            TBLK[(L, sc, pair)] = pd.Series(d[m]).groupby(F["bk"][m]).mean()
    print(f"{L}: rows {len(y)} | cert records {len(set(F['key'][cert]))} | final plan vs truth {plan_vs_truth(F)} | O1 {O[True][3]} anchors {len(O[True][0])} chains {len(O[True][1])} | O2 {O[False][3]} anchors {len(O[False][0])} chains {len(O[False][1])} | final anchors {len(F['A_tag'])} chains {len(F['CHfull'])}", flush=True)
T = pd.DataFrame(TROWS)
# ---------------- EC ----------------
__file__ = os.path.join(RES, "ac1_anchor_coverage_ec_v1.py")
src = open(os.path.join(RES, "ac1_anchor_coverage_ec_v1.py"), encoding="utf-8").read()
exec(compile(src[src.index("import os, sys, pickle"):src.index("ARMS = {")], "ac1pre", "exec"))
__file__ = os.path.join(RES, "abl3_three_conditions_v1.py")
_e = open(os.path.join(RES, "ect2_ec_field_agreement_anchor_v1.py"), encoding="utf-8").read()
exec(compile(_e[_e.index("import importlib.util"):_e.index("\nARMS = {")], "ect2pre", "exec"))
E2 = {**A8, **NEW}
R = pd.read_csv(os.path.join(RES, "nh1_rows_v1.csv.gz"))
MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); mt = np.array([MT[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
FOLDREC = {vk: set(G.rec) for vk, G in R.groupby(["validator", "fold"])}
def ec_plan(v, k, r, arm):
    if arm == "E2" or r not in TRUEP:
        if r not in E2: return None
        p, q = E2[r]; return None if p in FOLDREC[(v, k)] else (p, q)
    p = TRUEP[r]
    if p in FOLDREC[(v, k)] or p not in idx or not TRAIN[idx[p]] or not np.isfinite(EC[idx[p], 23]): return None
    if arm == "O1" and not abs(TIN.at[r, 0] - TIN.at[p, 23]) <= 1.5: return None
    return (p, 1.0)
ECNT = {}
for s in SEEDS:
    R[f"E0_{s}"] = (R[f"A1_{s}"] - .1 * mt) / .9
    for arm in ("E2", "O1", "O2"):
        out = R[f"E0_{s}"].values.copy()
        for (v, k, r), g in R.groupby(["validator", "fold", "rec"]):
            pl = ec_plan(v, k, r, arm)
            if s == SEEDS[0] and r in TRUEP:
                e2 = ec_plan(v, k, r, "E2"); key_ = (v, arm)
                ECNT.setdefault(key_, {"same": 0, "diff": 0, "added": 0, "dropped": 0, "none": 0})
                ECNT[key_]["none" if pl is None and e2 is None else "added" if e2 is None else "dropped" if pl is None else "same" if pl[0] == e2[0] else "diff"] += 1
            if pl is None: continue
            p, q = pl
            ii = g.sort_values("hour").index.values; x = R.loc[ii, f"E0_{s}"].values; pm = np.cumsum(x) / np.arange(1, len(x) + 1)
            out[ii] = x + .8 * q * (EC[idx[p], 23] - pm)
        R[f"{ {'E2': 'C2', 'O1': 'O1', 'O2': 'O2'}[arm] }_{s}"] = out
    R[f"C3_{s}"] = R[f"E0_{s}"]
EROWS = []; EBLK = {}
for v in ("DIAG10", "EL1", "P2LOO"):
    g = R[R.validator == v]; y = g.y.values; p2 = g.p2.values.astype(bool); cert = g.rec.isin(list(TRUEP)).values
    key = (g.farm + "_" + (g.day // 5).astype(str)).values
    scopes = (("all", np.ones(len(y), bool)), ("p2", p2), ("cert", cert), ("cert_p2", cert & p2))
    for a in ("C3", "C2", "O1", "O2"):
        for sc, m in scopes:
            EROWS.append(dict(target="ec", val=v, arm=a, scope=sc, n=int(m.sum()), rmse=float(np.mean([rmse((g[f"{a}_{s}"].values - y)[m]) for s in SEEDS])) if m.any() else np.nan))
    for sc, m in scopes:
        if not m.any(): continue
        for pair in (("O1", "C2"), ("C2", "C3"), ("O1", "C3")):
            d = np.mean([(g[f"{pair[0]}_{s}"].values - y) ** 2 - (g[f"{pair[1]}_{s}"].values - y) ** 2 for s in SEEDS], 0)
            EBLK[(v, sc, pair)] = pd.Series(d[m]).groupby(key[m]).mean()
    print(f"EC {v}: rows {len(y)} (2nd {int(p2.sum())}, cert rows {int(cert.sum())}) | cert-record plan vs final O1 {ECNT.get((v, 'O1'))} O2 {ECNT.get((v, 'O2'))}", flush=True)
E = pd.DataFrame(EROWS)
# ---------------- 보고 ----------------
ALL = pd.concat([T, E])
ALL["vs_C2_pct"] = [100 * (r.rmse / ALL[(ALL.target == r.target) & (ALL.val == r.val) & (ALL.scope == r.scope) & (ALL.arm == "C2")].rmse.iloc[0] - 1) for r in ALL.itertuples()]
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500); print(ALL.round(4).to_string(index=False))
rng = np.random.default_rng(20271103)
def bootP(BLK, keys):
    U = pd.concat([BLK[k] for k in keys if k in BLK], axis=1).mean(axis=1).values; ix = rng.integers(0, len(U), (20000, len(U)))
    return float((U[ix].mean(1) >= 0).mean()), len(U)
for sc in ("all", "p2", "cert", "cert_p2"):
    for pair in (("O1", "C2"), ("C2", "C3"), ("O1", "C3")):
        pt, nt = bootP(TBLK, [(L, sc, pair) for L in FR]); pe, ne = bootP(EBLK, [(v, sc, pair) for v in ("DIAG10", "EL1", "P2LOO")])
        print(f"[{sc}] P({pair[0]} 가 {pair[1]} 보다 나쁘거나 같음, 블록 부트스트랩) temp {pt:.4f} (blocks {nt}) | EC {pe:.4f} (blocks {ne})")
os.makedirs(OUTD, exist_ok=True); out = os.path.join(OUTD, "abl3_three_conditions_v1.csv"); ALL.to_csv(out, index=False); print("saved", out)
