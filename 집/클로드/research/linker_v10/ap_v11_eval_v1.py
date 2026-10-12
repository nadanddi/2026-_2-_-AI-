# AP11 (2026-10-12 집 클로드) 평가 60 기록 v11 짝(0시, 전체 학습으로 학습) + TD2 층 분류(서술).
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

w_all = fit_clogit([(d[F11].values.astype(float), y) for b_, a_, d, y in TRG], l2=0.5, iters=300)
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig").set_index("row")
v7a = set(E7[(E7.p_h0 >= .8) & E7.prev_is_labelled.astype(bool) & (E7.dT_air.abs() <= 1.5)].index)
rows = []
for b in [i for i in range(N) if not TRAIN[i] and FARM[i] in ("F13", "F47")]:
    D = feats11(b, TRAIN.copy(), set())
    if len(D) == 0: continue
    a = D.a.values.astype(int); sc = D[F11].values.astype(float) @ w_all; p = np.exp(sc - sc.max()); p /= p.sum(); k = int(np.argmax(p)); pk = a[k]
    r = rid(b); dT = float(IN_[b, 0, 0] - IN_[pk, 23, 0]); lab = bool(TRAIN[pk])
    if r in v7a: tier = "v7 anchor (kept)"
    elif lab and p[k] >= .6 and abs(dT) <= 1.5: tier = "NEW certain"
    elif lab and p[k] >= .6: tier = "NEW gate-released"
    else: tier = ""
    rows.append(dict(rec=r, v7=E7.at[r, "prev_h0"], v11=rid(pk), p=round(float(p[k]), 3), labelled=lab, dT=round(dT, 1), tier=tier))
T = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(T.tier.value_counts().to_string()); print(T[T.tier.str.startswith("NEW")].to_string(index=False))
print("v7 anchors where v11 picks a different predecessor:", T[(T.tier == "v7 anchor (kept)") & (T.v7 != T.v11)][["rec", "v7", "v11", "p"]].to_string(index=False))
T.to_csv(os.path.join(env.LOCAL, "linker_v10", "eval60_v11.csv"), index=False)
