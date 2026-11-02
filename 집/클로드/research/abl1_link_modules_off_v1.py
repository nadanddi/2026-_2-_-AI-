# -*- coding: utf-8 -*-
"""ABL1 (2026-11-02 집 클로드; 사용자 '지금 돌려' — 발표 자기 점검 3번·공통 1: 정형 최종 답안(submission_35e)에서
대회 자료 규칙성에 기대는 '어제 이어 붙이기'(FL 링커 -> 어제 정답 앵커·예측 사슬)를 끈 성능).
서술용(채택 판정 아님). 사전 고정(커밋 후 실행). 학습·평가 예측·제출물 변경 없음.

온도: TXP2(7.055) 틀 그대로 = 34 구조(33 + CHT) = 35e 온도와 같은 구조. 검증기 EXT10·EXT12·DIAG10, 2차 행(일 >= 179)·전체 행.
  A0 NX          : 기본 예측만(cellm 대신 셀별 B2.w40 x)
  A1 NX+TX3      : + TX3 (일반 방법만 = '규칙성 기댄 부분을 뺀' 구성)
  A2 +앵커       : + 어제 정답 앵커(p >= .8 + TAG2)
  A3 최종(35e)   : + 예측 사슬(CH5 + CHT)  = txp2 의 기준 B 와 같은 값(검산)
EC: ECT2(7.044) 틀 그대로(nh1_rows: DIAG10·EL1·P2LOO, 시드 1357·2468·3579), 2차 행·전체 행.
  E0 NH          : pre + HR5 (앵커 없음 = 일반 방법만)
  E1 +앵커 p>=.8 : 29회차 EC 앵커
  E2 최종(35e)   : + TAG2 EC 앵커 (= ECT2 안)
보고: 검증기·안별 RMSE(셀 평균), 최종 대비 %, A1 vs A3 · E0 vs E2 블록 부트스트랩 P(A1 이 최종보다 나쁠 확률이 아니라 '최종이 더 나은 정도'의 서술), 바뀐 기록 수.
주의(서술에 명시): 계절 지표(외기 쌍둥이로 2차 날짜 복원)는 모든 안에 남음 / 검증기는 학습 기록을 가린 것(평가 자료 아님) / 링커 가중은 같은 학습 자료로 맞춤.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u abl1_link_modules_off_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
# ---------------- 온도 ----------------
_src = open(os.path.join(HERE, "txp2_tx3_pass2_v1.py"), encoding="utf-8").read()
_src = _src[:_src.index("# ---- TXP2:")]
exec(compile(_src, "txp2pre", "exec"))
__file__ = os.path.join(HERE, "abl1_link_modules_off_v1.py")
rmse = lambda e: float(np.sqrt(np.nanmean(np.square(e))))
TROWS = []; TBLK = {}
for L, F in FR.items():
    y = F["y"]; zero = np.zeros(len(y)); p2 = F["M"].day.values >= 179; ok = np.isfinite(y)
    P = {a: [] for a in ("A0", "A1", "A2", "A3")}
    for b in F["bs"]:
        for fam in "AB":
            x = B2.w40(F["M"], b, F["cx"], fam)
            P["A0"].append(x)
            P["A1"].append(x + F["TXo"])
            P["A2"].append(assemble2(x, F["A_tag"], {}, F["pos"], F["h"], F["TXo"], F["order"], .9, 8.0, {}))
            P["A3"].append(assemble2(x, F["A_tag"], F["CHfull"], F["pos"], F["h"], F["TXo"], F["order"], .9, 8.0, {}))
    for a in P:
        for sc, m in (("p2", p2 & ok), ("all", ok)):
            TROWS.append(dict(target="temp", val=L, arm=a, scope=sc, rmse=float(np.mean([rmse((p - y)[m]) for p in P[a]]))))
    nan = sum(1 for r in F["A_tag"]); nch = len([r for r in F["CHfull"] if r not in F["A_tag"]])
    rp2 = set(F["key"][p2])
    print(f"{L}: rows {len(y)} (2nd {int(p2.sum())}) | records {len(F['pos'])} (2nd {len(rp2)}) | anchored {nan} (2nd {len([r for r in F['A_tag'] if r in rp2])}) | chained {nch} (2nd {len([r for r in F['CHfull'] if r not in F['A_tag'] and r in rp2])})", flush=True)
    for sc, m in (("p2", p2 & ok), ("all", ok)):
        d = np.mean([(p - y) ** 2 - (q - y) ** 2 for p, q in zip(P["A1"], P["A3"])], 0)
        TBLK[(L, sc)] = pd.Series(d[m]).groupby(F["bk"][m]).mean()
T = pd.DataFrame(TROWS)
# ---------------- EC ----------------
src = open(os.path.join(HERE, "ac1_anchor_coverage_ec_v1.py"), encoding="utf-8").read()
exec(compile(src[src.index("import os, sys, pickle"):src.index("ARMS = {")], "ac1pre", "exec"))
__file__ = os.path.join(HERE, "abl1_link_modules_off_v1.py")
_e = open(os.path.join(HERE, "ect2_ec_field_agreement_anchor_v1.py"), encoding="utf-8").read()
exec(compile(_e[_e.index("import importlib.util"):_e.index("\nARMS = {")], "ect2pre", "exec"))
ARMS = {"E1": A8, "E2": {**A8, **NEW}}
R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz"))
MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); mt = np.array([MT[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
FOLDREC = {vk: set(G.rec) for vk, G in R.groupby(["validator", "fold"])}
for s in SEEDS:
    R[f"E0_{s}"] = (R[f"A1_{s}"] - .1 * mt) / .9
    for name, AN in ARMS.items():
        out = R[f"E0_{s}"].values.copy()
        for (v, k, r), g in R.groupby(["validator", "fold", "rec"]):
            if r not in AN: continue
            p, q = AN[r]
            if p in FOLDREC[(v, k)]: continue
            ii = g.sort_values("hour").index.values; x = R.loc[ii, f"E0_{s}"].values; pm = np.cumsum(x) / np.arange(1, len(x) + 1)
            out[ii] = x + .8 * q * (EC[idx[p], 23] - pm)
        R[f"{name}_{s}"] = out
EROWS = []; EBLK = {}
for v in ("DIAG10", "EL1", "P2LOO"):
    g = R[R.validator == v]; y = g.y.values; p2 = g.p2.values.astype(bool); key = (g.farm + "_" + (g.day // 5).astype(str)).values
    for a in ("E0", "E1", "E2"):
        for sc, m in (("p2", p2), ("all", np.ones(len(y), bool))):
            EROWS.append(dict(target="ec", val=v, arm=a, scope=sc, rmse=float(np.mean([rmse((g[f"{a}_{s}"].values - y)[m]) for s in SEEDS]))))
    for sc, m in (("p2", p2), ("all", np.ones(len(y), bool))):
        d = np.mean([(g[f"E0_{s}"].values - y) ** 2 - (g[f"E2_{s}"].values - y) ** 2 for s in SEEDS], 0)
        EBLK[(v, sc)] = pd.Series(d[m]).groupby(key[m]).mean()
    nanc = len({r for r in set(g.rec) if r in ARMS["E2"]}); print(f"EC {v}: rows {len(y)} (2nd {int(p2.sum())}) | records with E2 anchor candidate {nanc}", flush=True)
E = pd.DataFrame(EROWS)
# ---------------- 보고 ----------------
ALL = pd.concat([T, E]); fin = {"temp": "A3", "ec": "E2"}
ALL["vs_final_pct"] = [100 * (r.rmse / ALL[(ALL.target == r.target) & (ALL.val == r.val) & (ALL.scope == r.scope) & (ALL.arm == fin[r.target])].rmse.iloc[0] - 1) for r in ALL.itertuples()]
pd.set_option("display.width", 200); print(ALL.round(4).to_string(index=False))
rng = np.random.default_rng(20271102)
def bootP(BLK, keys):
    U = pd.concat([BLK[k] for k in keys], axis=1).mean(axis=1).values; idx_ = rng.integers(0, len(U), (20000, len(U)))
    return float((U[idx_].mean(1) <= 0).mean()), len(U)
for sc in ("p2", "all"):
    pt, nt = bootP(TBLK, [(L, sc) for L in FR]); pe, ne = bootP(EBLK, [(v, sc) for v in ("DIAG10", "EL1", "P2LOO")])
    print(f"[{sc}] P(끈 판이 최종보다 나쁘지 않음) temp A1 vs A3 {pt:.4f} (blocks {nt}) | EC E0 vs E2 {pe:.4f} (blocks {ne})")
out = os.path.join(env.LOCAL, "abl1_link_modules_off_v1.csv"); ALL.to_csv(out, index=False); print("saved", out)
