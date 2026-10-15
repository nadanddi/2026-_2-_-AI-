# -*- coding: utf-8 -*-
"""PCH1 (2026-10-16 집 클로드; 주최 측 답변 2번(6.978) '같은 온실 앞 번호 평가 기록에 고정 규칙이 낸 예측값을 뒤 기록 입력으로 - 허용' -> 사용자 '진행해').
사전 고정(커밋 후 실행).  온도.  예측 사슬 앵커.
기준 = 셀 -> V3LF 30% 8/10 -> FL K8(정답 있는 전날, 폴드 학습 쪽) -> TX3(.25/10/.8)  (CVJ 기준).
사슬: K8 이 없는 기록 r 중 FL 링크 전날 p 가 (확신 >= .8, 같은 농장, 앞 날짜, 같은 폴드 검증 기록 = 정답 없음, |r 0시 실내 - p 23시 실내| <= 1.5) 이면
  r 의 0..23시에 strength * BETA[h] * clip(A - BIAS - x0, +-2) 더함,  A = p 의 최종 예측(앵커·TX3·사슬 반영 뒤) 23시 값, x0 = r 의 앵커 전 0시 예측.
  농장·날짜 순서로 처리(사슬의 사슬 허용).  BETA·BIAS = K8 과 같은 값(새로 맞추지 않음).
안: CH1 strength 1 | CH5 strength .5  (본페로니 2 -> alpha .0125).
판정: CV8·N·CK2 12셀 모두 개선 & 합친 5일 블록 P(악화) < .0125 & EL1 안전.  서술: 사슬 기록 수, 사슬 행·추운 사슬 행·0~5시, 평가 입력·링크만으로 센 사슬 대상 기록 수.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u pch1_prediction_chain_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "pch1_prediction_chain_v1.py")
import common, importlib.util as _iu
_s = _iu.spec_from_file_location("cv8", os.path.join(HERE, "cv8_cold_validator_v1.py")); C8 = _iu.module_from_spec(_s); _s.loader.exec_module(C8)
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); FLK = {r: (p, q) for r, p, q in zip(FLT.rec, FLT.prev, FLT.p)}
_wd = W.worlds(); WB = np.asarray(_wd[4], float); ELF = dict(_wd[7])["EL1"]; CKR = os.path.join(env.LOCAL, "yl1_ref_ckpt")
FE = ["in_temp", "ph_in_temp_1", "ph_in_temp_3", "ph_in_temp_8", "ph_in_temp_24", "out_temp", "ph_rad_2", "act_heating", "act_thermal", "hr_sin", "hr_cos"]
L0 = lab.copy()
if "hr_sin" not in L0: L0["hr_sin"], L0["hr_cos"] = np.sin(2 * np.pi * L0.hour / 24), np.cos(2 * np.pi * L0.hour / 24)
es = lambda T: .6108 * np.exp(17.27 * T / (T + 237.3)); e_in = es(L0.in_temp) * L0.in_hum / 100
L0["ah"] = 2167 * e_in / (L0.in_temp + 273.15); w_ = .622 * e_in / (101.325 - e_in); L0["enth"] = 1.006 * L0.in_temp + w_ * (2501 + 1.86 * L0.in_temp)
L0["dewgap"] = L0.in_temp - (237.3 * np.log(e_in.clip(1e-3) / .6108) / (17.27 - np.log(e_in.clip(1e-3) / .6108)))
PSY = ["ah", "enth", "dewgap"]
def spl_fit_predict(tm, vm, feats):
    Xt = L0.loc[tm, feats].to_numpy(float); Xv = L0.loc[vm, feats].to_numpy(float); med = np.nanmedian(Xt, 0)
    Xt = np.where(np.isnan(Xt), med, Xt); Xv = np.where(np.isnan(Xv), med, Xv)
    m = make_pipeline(SplineTransformer(n_knots=5, degree=3, extrapolation="linear"), StandardScaler(), Ridge(alpha=10.0)); m.fit(Xt, L0.sub_temp.values[tm], ridge__sample_weight=WB[tm])
    return pd.Series(m.predict(Xv), index=L0.row_id.values[vm])
def gate(M, lo_g, lo_h):
    g_ = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - lo_g) / 2, 0, 1)); order = M.sort_values(["farm", "day", "hour"]).index
    mm = M.loc[order].groupby(["farm", "day"]).in_temp.cummin().reindex(M.index); h_ = np.where(mm.isna(), 1, np.clip((mm - lo_h) / 2, 0, 1)); return g_ * h_
def tx3(M, key, c, T0, e0):
    cm = np.nan_to_num(np.array([CM.get((k, int(hh)), np.nan) for k, hh in zip(key, M.hour.values)]), nan=0)
    return c * np.clip((cm - e0) / .4, 0, 1) * np.nan_to_num(np.clip((T0 - M.in_temp.values) / 2, 0, 1), nan=0.0)
OPTS = {"CH1": 1.0, "CH5": .5}
RES, CNT = {}, []
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks),
                                ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(ckg, f)) for f in sorted(os.listdir(ckg)) if f.startswith(L + "_")], ignore_index=True)
    M = M.merge(Gd, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    v = M.V3LF.values; ok = ~np.isnan(v); vv = np.nan_to_num(v); TXo = tx3(M, key, .25, 10, .8); G8 = gate(M, 8, 10); t = M.in_temp.values
    TRS, VFOLD = {}, {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format)); vs = set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format))
        for r in vs: TRS[r] = trs; VFOLD[r] = k
    AF = {r: float(ST[IDX[FLK[r][0]], 23]) for r in recs if r in FLK and r in TRS and FLK[r][1] >= .8 and FLK[r][0] in TRS[r] and np.isfinite(ST[IDX[FLK[r][0]], 23])
          and abs(IN_[IDX[r], 0, 0] - IN_[IDX[FLK[r][0]], 23, 0]) <= 1.5}
    CH = {}
    for r in recs:
        if r in AF or r not in FLK or not isinstance(FLK[r][0], str): continue
        p, pq = FLK[r]
        if pq < .8 or p[:3] != r[:3] or int(p[4:7]) >= int(r[4:7]) or p not in pos or VFOLD.get(p) != VFOLD.get(r): continue
        if p in TRS.get(r, set()): continue
        if abs(IN_[IDX[r], 0, 0] - IN_[IDX[p], 23, 0]) > 1.5 or not (h[pos[p]] == 23).any(): continue
        CH[r] = p
    order = sorted(recs, key=lambda r: (r[:3], int(r[4:7])))
    def run(x, strength):
        out = x.copy()
        for r, a in AF.items():
            ix = pos[r]; x0 = x[ix][h[ix] == 0]
            if len(x0): out[ix] += BETA[h[ix]] * np.clip(a - BIAS - x0[0], -2, 2)
        out = out + TXo
        if strength > 0:
            for r in order:
                if r not in CH: continue
                ix = pos[r]; ip = pos[CH[r]]; x0 = x[ix][h[ix] == 0]; a = out[ip][h[ip] == 23]
                if len(x0) and len(a): out[ix] += strength * BETA[h[ix]] * np.clip(a[0] - BIAS - x0[0], -2, 2)
        return out
    P = {"B": []}; P.update({o: [] for o in OPTS})
    for s in O1.SEEDS:
        for fam in "AB":
            q = B.w40(M, "base_R_%d" % s, "codex_R", fam); q = q + .30 * G8 * ok * (vv - q)
            P["B"].append(run(q, 0)); [P[o].append(run(q, st)) for o, st in OPTS.items()]
    chm = np.isin(key, list(CH)); dmin = pd.Series(t).groupby(key).transform("min").values
    RES[L] = dict(P=P, y=y, t=t, ch=chm, h=h, bk=(L + "_" + M.farm + "_" + (M.day // 5).astype(str)).values)
    CNT.append(dict(layout=L, records=len(recs), labelled_anchor=len(AF), chain=len(CH), chain_cold_recs=int(len({r for r in CH if (dmin[pos[r]] <= 8).any()})), chain_rows=int(chm.sum())))
    print(L, "done", flush=True)
pd.set_option("display.width", 220); print(pd.DataFrame(CNT).to_string(index=False))
def cellch(L, c, m=None):
    R_ = RES[L]; y = R_["y"]; m = np.ones(len(y), bool) if m is None else m
    return [rm((p - y)[m]) / rm((b - y)[m]) - 1 for p, b in zip(R_["P"][c], R_["P"]["B"])] if m.any() else [np.nan]
rng = np.random.default_rng(20270151)
for c in OPTS:
    for L in RES:
        R_ = RES[L]; tt = np.nan_to_num(R_["t"], nan=99); chm = R_["ch"]
        print(f"  {c} {L}: cells {' '.join('%+.3f' % (100 * x) for x in cellch(L, c))} | chained rows {100 * np.mean(cellch(L, c, chm)):+.2f}% | chained <=8C {100 * np.mean(cellch(L, c, chm & (tt <= 8))):+.2f}% | chained hours 0-5 {100 * np.mean(cellch(L, c, chm & (R_['h'] <= 5))):+.2f}%")
    cells = cellch("CV8", c) + cellch("N", c) + cellch("CK2", c)
    dd = pd.concat([pd.Series(np.mean([(p - RES[L]["y"]) ** 2 - (b - RES[L]["y"]) ** 2 for p, b in zip(RES[L]["P"][c], RES[L]["P"]["B"])], 0)).groupby(RES[L]["bk"]).mean() for L in ("CV8", "N", "CK2")]).values
    pw = float((dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1) >= 0).mean())
    R_ = RES["EL1"]; c2 = cellch("EL1", c); d2 = pd.Series(np.mean([(p - R_["y"]) ** 2 - (b - R_["y"]) ** 2 for p, b in zip(R_["P"][c], R_["P"]["B"])], 0)).groupby(R_["bk"]).mean().values
    pb = float((d2[rng.integers(0, len(d2), (20000, len(d2)))].mean(1) < 0).mean()); safe = not (np.mean(c2) > 0 and pb < .025)
    print(f"PCH1 {c}: 12 cells all better {all(x < 0 for x in cells)} | combined block P(worse) {pw:.4f} (alpha .0125) | EL1 {100 * np.mean(c2):+.3f}% safe {safe} -> {'PASS' if all(x < 0 for x in cells) and pw < .0125 and safe else 'FAIL'}")
EV = pd.read_csv(os.path.join(env.LOCAL, "fl3_eval_links.csv")); TX = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id", "in_temp"]); TX["rec"] = TX.row_id.str[:7]
dm = TX.groupby("rec").in_temp.min(); EV["same_earlier"] = [isinstance(p, str) and p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) for r, p in zip(EV.rec, EV.prev)]
EV["chain"] = EV.same_earlier & ~EV.prev_is_labelled & (EV.p >= .8) & (EV.dT_air.abs() <= 1.5); EV["cold"] = EV.rec.map(dm) <= 8
print(f"  eval (inputs/links only): records {len(EV)} | chain-eligible {int(EV.chain.sum())} | of which cold (day min <= 8) {int((EV.chain & EV.cold).sum())} | labelled K8-type {int(((EV.p >= .8) & EV.prev_is_labelled & (EV.dT_air.abs() <= 1.5)).sum())}")
