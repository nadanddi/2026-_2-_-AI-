# EV11c (2026-10-12 집 클로드; 사용자: '앵커가 바뀐 기록만 따로 봐봐') 서술(판정 아님): EV11 두 안 x 4칸, 앵커가 바뀐 기록만의 RMSE 변화·좋아진 비율, 1·2차, 층(λ1/λ.5)별.
# -*- coding: utf-8 -*-
"""EV11 (2026-10-12 집 클로드; 사용자: 'v11 + 확신 문턱으로 온도 검증 채점').  사전 고정(이 파일 커밋 후 실행).  온도만.
v11 거울 CV 선택(기본 기하, 덩어리마다 재학습) — 정답 기록을 고른 경우만.  층: 확실(λ 1) = v11 확신 ≥ .6 & |dT_air| ≤ 1.5, 애매(λ .5) = 그 밖(확신 < .6 또는 차단).
v7 앵커(13회차, p ≥ .8·정답·|dT| ≤ 1.5)는 λ 1 로 포함. ARM1 = v7 우선(같은 기록에서 v7 이 있으면 v7 의 짝), ARM2 = v11 우선(v11 확실이면 v11 짝, 아니면 v7).
식 = 13회차: q18_h + λ·BETA_h·clip(A − .36 − q0, ±2), 전날이 EV 폴드 val/del 이면 버림. 기준 = v22.
판정(본페로니 2안 α .0125): 4칸 모두 전체 개선 AND 2차 개선(또는 같음) AND 기록 부트스트랩 P(악화) < .0125 AND 상위 1기록 몫 ≤ .5.
"""
# -*- coding: utf-8 -*-
"""V11 (2026-10-12 집 클로드; 사용자: '같은 ID 에 가중을 주면 안 되겠다').  서술(ID3 와 같은 10기하·2차·깨끗한 정답).
짝찾기 v11 = v10 후보·가시성 규칙(규정) 그대로, 특징에서 ID·번호 기반(isprev, dsdev, dsna, d1, d2w, d2c, d3, ldn, xf, lat) 전부 제거,
  대신 모든 후보 공통 달력 특징: cd1 = [후보 날짜 == 대상 날짜 − 1], cdev = min(|후보 날짜 − (대상 날짜 − 1)|, 6)/6, cna = [대상 또는 후보 날짜 모름].
  날짜: 대상 = 0시 외기 4칸이 정확히 같은 '보이는 정답 기록'들의 DM11 날짜(최빈); 후보 = 보이는 정답 기록이면 DM11, 아니면 같은 외기 일치 방식.
  나머지 특징: oj, v0..v7, fp(0시 지문), excl(다음날이 이미 정답으로 확정된 동).
비교: v10+fp(ID 특징 포함), C1(HP2/v10), C1'(HP2/v11).  같은 ID·다른 ID 진실별.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
exec(open("id1_input_only_linker_v1.py", encoding="utf-8").read().split("TRG = []")[0])
SUCC = {}
for a_, b_, c_ in zip(TLa.a1, TLa.b, TLa.cert):
    if c_: SUCC.setdefault(int(a_), []).append(int(b_))
F11 = ["oj"] + [f"v{j}" for j in range(8)] + ["fp", "excl", "cd1", "cdev", "cna"]
def date_of(i, lab):
    if lab[i] and np.isfinite(DAY[i]): return DAY[i]
    vi = np.where(lab)[0]; eq = np.all(np.abs(X[vi, 0] - X[i, 0]) < 1e-6, axis=1); ds = DAY[vi[eq]]; ds = ds[np.isfinite(ds)]
    return pd.Series(ds).mode().iloc[0] if len(ds) else np.nan
def feats11(b, lab, removed):
    D = feats(b, lab, removed)
    if len(D) == 0: return D
    db = date_of(b, lab & (np.arange(N) != b)); a = D.a.values.astype(int); da = np.array([date_of(x, lab) for x in a])
    na = ~np.isfinite(da) | (not np.isfinite(db))
    D["cd1"] = np.where(na, 0.0, (da == db - 1).astype(float)); D["cdev"] = np.where(na, 0.0, np.minimum(np.abs(da - (db - 1)), 6) / 6); D["cna"] = na.astype(float)
    return D
TRG = []
for b, a in zip(TL7.b, TL7.a):
    lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
    df = feats11(b, lab, TEST_IDX)
    if len(df) == 0: continue
    y = np.where(df.a.values == a)[0]
    if len(y): TRG.append((b, a, df, y[0]))

H_ = os.path.join(env.ROOT, "집", "클로드", "research")
picks = {}
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist()); rem2 = rem | oth
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        w11 = fit_clogit([(d[F11].values.astype(float), y) for b_, a_, d, y in TRG if not touch[b_] and not touch[a_]], l2=0.5, iters=300)
        lab = TRAIN.copy(); lab[touch] = False
        for b in blk:
            if not TRAIN[b]: continue
            D = feats11(b, lab, rem2)
            if len(D) == 0: continue
            a = D.a.values.astype(int); sc = D[F11].values.astype(float) @ w11; p = np.exp(sc - sc.max()); p /= p.sum(); k = int(np.argmax(p))
            if lab[a[k]]: picks[rid(b)] = (rid(a[k]), float(p[k]), float(IN_[b, 0, 0] - IN_[a[k], 23, 0]))
pd.DataFrame([(r, *v) for r, v in picks.items()], columns=["rec", "prev", "p", "dT"]).to_csv(os.path.join(env.LOCAL, "linker_v10", "v11_mirror_picks.csv"), index=False)
V = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig").set_index("rec")
v7 = {r: V.at[r, "prev"] for r in V.index[(V.p >= .8) & V.prev_is_labelled.astype(bool) & (V.dT_air.abs() <= 1.5)]}
def tiers(v7_first):
    W = {}
    for r, (pv, p, dT) in picks.items():
        w = 1.0 if (p >= .6 and abs(dT) <= 1.5) else .5
        W[r] = (pv, w)
    for r, pv in v7.items():
        if v7_first or r not in W or W[r][1] < 1.0: W[r] = (pv, 1.0)
    return W
ARMS = {"ARM1_v7first": tiers(True), "ARM2_v11first": tiers(False)}
for k, W in ARMS.items(): print(k, "full", sum(1 for v in W.values() if v[1] == 1), "half", sum(1 for v in W.values() if v[1] < 1))
idx = {rid(i): i for i in range(N)}
FO = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
DD = pd.read_pickle(os.path.join(H_, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); DD["rec"] = DD.row_id.str[:7]; DD = DD.rename(columns={"sub_temp": "y"}).reset_index(drop=True)
BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
rng = np.random.default_rng(20261019); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))


rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))
out = []
for arm, W in ARMS.items():
    for c in ["7A", "7B", "101A", "101B"]:
        q18 = DD[f"v18_a_{c}"].values; base = DD[f"v22_a_{c}"].values; y = DD.y.values; new = q18.copy(); tier = {}
        for rec, G in DD.groupby("rec"):
            if rec not in W: continue
            pv, w = W[rec]; k = int(G.fold.iloc[0])
            if pv in hid[k]: continue
            ix = G.index.values; hh = G.hour.values; q0 = q18[ix[hh == 0]][0]
            new[ix] = q18[ix] + w * BETA[hh] * np.clip(ST[idx[pv], 23] - .36 - q0, -2, 2); tier[rec] = w
        chg = DD.rec.map(lambda r: abs(0) ).values
        rows_ch = pd.Series(np.abs(new - base) > 1e-9).groupby(DD.rec.values).any()
        chset = set(rows_ch[rows_ch].index)
        for part, m in (("all changed", DD.rec.isin(chset).values), ("changed pass-1", DD.rec.isin(chset).values & (DD.day.values < 179)), ("changed pass-2", DD.rec.isin(chset).values & (DD.day.values >= 179)),
                        ("changed λ1", DD.rec.isin([r for r in chset if tier.get(r) == 1.0]).values), ("changed λ.5", DD.rec.isin([r for r in chset if tier.get(r) == .5]).values)):
            if m.sum() == 0: continue
            per = pd.DataFrame(dict(r=DD.rec.values[m], b=(base[m] - y[m]) ** 2, n=(new[m] - y[m]) ** 2)).groupby("r").sum()
            out.append(dict(arm=arm, cell=c, part=part, records=len(per), rmse_base=rmse(base[m] - y[m]), rmse_new=rmse(new[m] - y[m]), pct=100 * (rmse(new[m] - y[m]) / rmse(base[m] - y[m]) - 1), improved=float((per.n < per.b).mean())))
O = pd.DataFrame(out); pd.set_option("display.width", 200)
print(O.groupby(["arm", "part"]).agg(records=("records", "first"), rmse_base=("rmse_base", "mean"), rmse_new=("rmse_new", "mean"), pct=("pct", "mean"), improved=("improved", "mean")).round(3).to_string())
