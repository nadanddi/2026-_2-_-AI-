# -*- coding: utf-8 -*-
"""CMP3b 진단(판정 아님, 2026-10-13 집 클로드, hx8/compress).
1) CMP2 의 KH(하루 전체 입력 이웃, 비인과) 이득이 CMP3 인과판에서 사라진 이유 분해:
   같은 계산을 (a) 하루 전체 거리·비인과 그대로, 풀 = DIAG10 고EC 기록(CMP2 와 같음) / 풀 = 고EC 학습 35,
   (b) 위에서 같은 날짜 기록만 풀에서 뺌, (c) 인과판(CMP3 KH5_P1). 같은 칸·같은 교차적합.
2) 사후(표를 본 뒤 떠올린 것 → 채택 불가, 서술만) SHC: '같은 ID 앞 기록 예측 사슬 C_pred >= 1' 일 때만 SH 보정.
   CMP2 표에서 헛경보 10 중 6 이 C_pred 0, 실제 고EC 30 중 10 이 C_pred 0 → 게이트로 헛경보를 거르면 SH 가 사나 보는 것.
계수는 모두 다른 에피소드에서만(사례 단위 교차적합). 평가 입력·예측에 맞추는 것 없음.
"""
import sys, os
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cm_common import *

TAU = 0.9; HIT = 1.0
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

pa = {int(b): int(a) for a, b, c in zip(T.a1, T.b, T.c1) if c <= 2.0}
O4 = np.nan_to_num(A[:, :, 3:7])
SD = np.zeros((N, N), bool)
for i in range(N):
    SD[i] = np.abs(O4 - O4[i]).mean((1, 2)) < 1e-6
np.fill_diagonal(SD, False)
trn = np.where(TRAIN)[0]
mu = np.nanmean(A[trn].reshape(-1, 14), 0); sdv = np.nanstd(A[trn].reshape(-1, 14), 0) + 1e-9
Z = np.nan_to_num((A - mu) / sdv)
S2 = Z.reshape(N, -1)
sq = (S2 ** 2).sum(1)
DF = np.maximum(sq[:, None] + sq[None, :] - 2 * S2 @ S2.T, 0)  # 하루 전체 거리(CMP2 와 같은 공간)

sgD, _ = oof("DIAG10"); sgE, _ = oof("EL1"); sgP, _ = oof("P2LOO")
pmall = {}
for k in r.rid[TRAIN]:
    v = sgD.get(k, sgE.get(k, sgP.get(k)))
    if v is not None: pmall[rid2i[k]] = float(np.mean(v))
Fset = {i for i in range(N) if TRAIN[i] and (ydm[i] >= HIT or pmall.get(i, 0) >= TAU)}
par = {i: i for i in range(N)}
def fd(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for b, a in pa.items():
    if b in Fset and a in Fset: par[fd(b)] = fd(a)
grp = np.array([fd(i) for i in range(N)])
HI35 = np.array([i for i in range(N) if TRAIN[i] and ydm[i] >= HIT])


def walk(i, flag, maxk=30):
    k, x = 0, i
    while k < maxk:
        p = pa.get(x)
        if p is None or not TRAIN[p]: return k
        f = flag(p)
        if f is None or not f: return k
        k += 1; x = p
    return k


def cases(col, seed):
    sg, _ = oof("DIAG10", col, seed)
    rows = []
    for k, v in sg.items():
        i = rid2i[k]
        if not TRAIN[i]: continue
        hp = np.asarray(v, float)
        if hp.mean() < TAU: continue
        rows.append(dict(i=i, rid=k, hp=hp, cm=np.cumsum(hp) / np.arange(1, 25), yh=Ytr[i], y=ydm[i], pm=hp.mean(),
                         C=walk(i, lambda p: (pmall[p] >= TAU) if p in pmall else None)))
    return rows, sg


def knn_full(i, poolset, excl_groups, drop_sd, K=5):
    m = np.array([(grp[c] not in excl_groups) and c != i and not (drop_sd and SD[i, c]) for c in poolset])
    cand = poolset[m]
    return ydm[cand[np.argsort(DF[i, cand])[:K]]].mean()


def kh_nc(rows, poolset, drop_sd):
    """CMP2 형 비인과 KH: pred = pm + w (knn − pm), 모든 시간에 같은 폭."""
    gg = np.array([grp[d["i"]] for d in rows]); preds = []
    for j, d in enumerate(rows):
        zs, ts = [], []
        for jj, dd in enumerate(rows):
            if gg[jj] == gg[j]: continue
            kn = knn_full(dd["i"], poolset, {gg[jj], gg[j]}, drop_sd)
            zs.append(kn - dd["pm"]); ts.append(dd["y"] - dd["pm"])
        zs, ts = np.array(zs), np.array(ts)
        w = float(np.clip((zs * ts).sum() / (zs * zs).sum(), 0, 1))
        kn = knn_full(d["i"], poolset, {gg[j]}, drop_sd)
        preds.append(d["pm"] + w * (kn - d["pm"]))
    return np.array(preds)


def shc(rows, gate_min):
    """인과 SH, 단 C_pred >= gate_min 인 기록만(그 외는 그대로)."""
    gg = np.array([grp[d["i"]] for d in rows]); preds = []
    for j, d in enumerate(rows):
        ts = [((dd["yh"] - dd["hp"])[dd["cm"] >= TAU]) for jj, dd in enumerate(rows)
              if gg[jj] != gg[j] and dd["C"] >= gate_min and (dd["cm"] >= TAU).any()]
        a = float(np.concatenate(ts).mean()) if ts else 0.0
        if d["C"] >= gate_min:
            preds.append(np.where(d["cm"] >= TAU, d["hp"] + a, d["hp"]).mean())
        else:
            preds.append(d["hp"].mean())
    return np.array(preds)


def boot_p(e_new, e_ref, gg, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(gg)
    dv = np.array([(e_new[gg == u] ** 2).sum() - (e_ref[gg == u] ** 2).sum() for u in ug])
    bs = np.array([dv[rng.integers(0, len(ug), len(ug))].sum() for _ in range(B)])
    return float((bs >= 0).mean())


res = []
for col in ["sg", "nosg"]:
    for seed in SEEDS:
        rows, sg = cases(col, seed)
        y = np.array([d["y"] for d in rows]); gg = np.array([grp[d["i"]] for d in rows]); hi = y >= HIT
        e0 = y - np.array([d["pm"] for d in rows])
        poolD = np.array([rid2i[k] for k in sg if TRAIN[rid2i[k]] and ydm[rid2i[k]] >= HIT])  # CMP2 풀(DIAG10 기록만)
        variants = {
            "NC_poolD_sd포함(CMP2형)": kh_nc(rows, poolD, False),
            "NC_poolD_sd제외": kh_nc(rows, poolD, True),
            "NC_pool35_sd포함": kh_nc(rows, HI35, False),
            "NC_pool35_sd제외": kh_nc(rows, HI35, True),
            "SHC_C>=1(사후)": shc(rows, 1),
            "SH_전부(인과)": shc(rows, 0),
        }
        for nm, pr in variants.items():
            e = y - pr
            res.append(dict(col=col, seed=seed, model=nm, rmse=rmse(e), m0=rmse(e0), hi=rmse(e[hi]), fp=rmse(e[~hi]),
                            impr=float(np.mean(np.abs(e) < np.abs(e0) - 1e-12)), P=boot_p(e, e0, gg)))
        if col == "sg" and seed == SEEDS[0]:
            C = np.array([d["C"] for d in rows])
            P(f"[sg 47] C_pred 0: 헛경보 {int(((C==0)&~hi).sum())}/{int((C==0).sum())}, C_pred>=1: 헛경보 {int(((C>=1)&~hi).sum())}/{int((C>=1).sum())}")
            # 같은 날짜 고EC 가 CMP2 형 이웃 1위였던 사례 수
            n1 = 0
            for d in rows:
                cand = np.array([c for c in poolD if grp[c] != grp[d["i"]] and c != d["i"]])
                if len(cand) and SD[d["i"], cand[np.argmin(DF[d["i"], cand])]]: n1 += 1
            P(f"[sg 47] CMP2 형 풀에서 이웃 1위가 같은 날짜 다른 동 고EC 인 사례 {n1}/{len(rows)}")

R = pd.DataFrame(res)
R.to_csv(os.path.join(HERE, "cmp3b_ablation_v1.csv"), index=False)
pd.set_option("display.width", 250)
g = R.groupby("model", sort=False).agg(rmse=("rmse", "mean"), m0=("m0", "mean"), hi=("hi", "mean"), fp=("fp", "mean"),
                                       impr=("impr", "mean"), Pmax=("P", "max"), Pmin=("P", "min"))
g["better_cells"] = R.assign(b=R.rmse < R.m0).groupby("model", sort=False).b.sum()
P("\n=== 하루 수준 RMSE (DIAG10 예측 고EC 학습 사례, 6칸 평균) ===")
P(g.round(4).to_string())

# 평가 8: 사후 SHC 참고치 (C 는 CMP2 의 같은 ID 번호 −2 사슬 C_pred)
rows, _ = cases("sg", None)
ts = [((d["yh"] - d["hp"])[d["cm"] >= TAU]) for d in rows if d["C"] >= 1 and (d["cm"] >= TAU).any()]
a = float(np.concatenate(ts).mean())
E2 = pd.read_csv(os.path.join(HERE, "cmp2_eval8_v1.csv"))
P(f"\n사후 SHC 계수 a {a:+.3f} (참고, 채택 불가). 평가 8: " + ", ".join(
    f"{rr.rec} {rr.sub22:.3f}→{(rr.sub22 + a) if rr.c >= 1 else rr.sub22:.3f}(C {rr.c})" for rr in E2.itertuples()))
open(os.path.join(HERE, "cmp3b_ablation_v1.log"), "w", encoding="utf-8").write("\n".join(out))
