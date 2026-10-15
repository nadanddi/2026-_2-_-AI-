# -*- coding: utf-8 -*-
"""MEM1 (2026-10-16 집 클로드; 사용자 '구성원 자체를 바꾸는 방향 진행해봐').  서술 선별(사전 고정, 커밋 후 실행).  온도.
배경: 비중·문턱 학습(STK1~4)은 닫힘; BMX1(7.013) 구성원 모양 오차 상관 .77~.83 -> 기존과 '다르게 틀리는' 새 구성원이 필요.
새 구성원 후보 4종(폴드마다 그 폴드 학습 행만으로 적합, 학습 가중 WB):
  GAMF 가법 스플라인(FE + 습공기, 직선 외삽, Ridge 10) 전 범위(CVJ 는 추운 행 게이트에서만 시험)
  LGBD LightGBM 목표 = 배지 - 3시간 지연 실내(ph_in_temp_3), 특징 FE + 습공기 + 실내 습도·CO2·외기 습도·일사·환기·순환팬
  ETD  ExtraTrees 같은 목표·특징(잎 20)          KNND 이웃 40(거리 가중, 표준화 FE + 습공기) 같은 목표 - 닮은 상황 평균
섞기: 셀(W40G-S) + w (z - 셀), 그 뒤 R27 + CH5 조립(FL7 거울 연결: K8 앵커·TX3·시간 감쇠 사슬 p >= .5) = 기준과 동일.
w 는 데이터가 정함: 0~.5(.05 간격) 중 '나머지 3배치' 평균 상대 MSE 최소 -> 남은 배치에 적용(배치 하나 빼기).
배치: CV8·N·CK2·EL1.  서술: 단독 RMSE, 셀과의 오차 상관(전체·모양·수준), 어제 정보 없는 기록·8도 이하 행 변화.
다음 단계 진출 기준(고정): 네 배치 모두 선택 w > 0 이고 RMSE 변화 < 0 인 후보만 MEM2(EXT10·EXT12·DIAG10 독립 판정)로.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u mem1_new_member_screen_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "mem1_new_member_screen_v1.py")
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
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
S1, TAU = .9, 8.0
def link_plan(LK, recs, pos, h, FOLDOF, lab_ok):
    AF, CH = {}, {}
    for r in recs:
        if r not in LK or not isinstance(LK[r][0], str): continue
        p, q = LK[r]
        if p not in IDX: continue
        dt = abs(IN_[IDX[r], 0, 0] - IN_[IDX[p], 23, 0])
        if not np.isfinite(dt) or dt > 1.5: continue
        if lab_ok(r, p):
            if q >= .8 and np.isfinite(ST[IDX[p], 23]): AF[r] = float(ST[IDX[p], 23])
        elif q >= .5 and p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) and p in pos and FOLDOF.get(p) == FOLDOF.get(r) and (h[pos[p]] == 23).any(): CH[r] = p
    return AF, CH
def assemble(x, AF, CH, pos, h, TXo, order):
    out = x.copy()
    for r, a in AF.items():
        ix = pos[r]; x0 = x[ix][h[ix] == 0]
        if len(x0): out[ix] += BETA[h[ix]] * np.clip(a - BIAS - x0[0], -2, 2)
    out = out + TXo
    for r in order:
        if r not in CH: continue
        ix = pos[r]; ip = pos[CH[r]]; x0 = x[ix][h[ix] == 0]; a = out[ip][h[ip] == 23]
        if len(x0) and len(a): out[ix] += S1 * np.exp(-h[ix] / TAU) * BETA[h[ix]] * np.clip(a[0] - BIAS - x0[0], -2, 2)
    return out
FX = FE + PSY + ["in_hum", "in_co2", "out_hum", "out_rad", "act_vent", "act_circfan"]
def cand_fit_predict(tm, vm, name):
    feats = FE + PSY if name in ("GAMF", "KNND") else FX
    Xt = L0.loc[tm, feats].to_numpy(float); Xv = L0.loc[vm, feats].to_numpy(float); med = np.nanmedian(Xt, 0); Xt = np.where(np.isnan(Xt), med, Xt); Xv = np.where(np.isnan(Xv), med, Xv)
    yt = L0.sub_temp.values[tm]; r3 = L0.ph_in_temp_3.values; ref = np.where(np.isnan(r3), L0.in_temp.values, r3); ref_t = ref[tm]; ref_v = ref[vm]
    ok = np.isfinite(yt) & np.isfinite(ref_t); wt = WB[tm]
    if name == "GAMF":
        m = make_pipeline(SplineTransformer(n_knots=5, degree=3, extrapolation="linear"), StandardScaler(), Ridge(alpha=10.0)); m.fit(Xt[ok], yt[ok], ridge__sample_weight=wt[ok]); pr = m.predict(Xv)
    elif name == "LGBD":
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=.03, num_leaves=31, min_child_samples=40, subsample=.8, subsample_freq=1, colsample_bytree=.8, random_state=11, verbose=-1, deterministic=True, force_col_wise=True)
        m.fit(Xt[ok], (yt - ref_t)[ok], sample_weight=wt[ok]); pr = ref_v + m.predict(Xv)
    elif name == "ETD":
        m = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=20, max_features=.6, n_jobs=-1, random_state=11); m.fit(Xt[ok], (yt - ref_t)[ok], sample_weight=wt[ok]); pr = ref_v + m.predict(Xv)
    else:
        m = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=40, weights="distance")); m.fit(Xt[ok], (yt - ref_t)[ok]); pr = ref_v + m.predict(Xv)
    return pd.Series(pr, index=L0.row_id.values[vm])
CANDS = ("GAMF", "LGBD", "ETD", "KNND"); WG = np.round(np.arange(0, .51, .05), 2)
MIR7 = pd.read_csv(os.path.join(env.LOCAL, "fl7_mirror_links.csv")); RES = {}
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks),
                                ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TXo = tx3(M, key, .25, 10, .8); TRS, VF = {}, {}; SP = {c: [] for c in CANDS}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs; VF[r] = k
        for c in CANDS: SP[c].append(cand_fit_predict(tm, vm, c))
    Z = {c: pd.concat(s_).groupby(level=0).first().reindex(M.row_id).values for c, s_ in SP.items()}
    LK = {r: (p, q) for r, p, q in zip(MIR7[MIR7.layout == L].rec, MIR7[MIR7.layout == L].prev, MIR7[MIR7.layout == L].p)}; order = sorted(recs, key=lambda r: (r[:3], int(r[4:7])))
    AF, CH = link_plan(LK, recs, pos, h, VF, lambda r, p: r in TRS and p in TRS[r]); linked = np.isin(key, list(AF) + list(CH))
    cells = [B.w40(M, "base_R_%d" % s, "codex_R", fam) for s in O1.SEEDS for fam in "AB"]
    base = [assemble(g, AF, CH, pos, h, TXo, order) for g in cells]
    P = {(c, w): [assemble(g + w * np.where(np.isfinite(Z[c]), Z[c] - g, 0.0), AF, CH, pos, h, TXo, order) for g in cells] for c in CANDS for w in WG if w > 0}
    RES[L] = dict(y=y, base=base, P=P, Z=Z, cell=np.mean(cells, 0), key=key, linked=linked, t=M.in_temp.values, bk=(M.farm + "_" + (M.day // 5).astype(str)).values)
    print(L, "rows", len(M), "| linked share", round(linked.mean(), 3), flush=True)
mse = lambda e: float(np.mean(np.square(e)))
print("\n== standalone & error correlation with cell (rows; within-record centred = shape; record means = level) ==")
for L, R_ in RES.items():
    y = R_["y"]; ec = R_["cell"] - y
    for c in CANDS:
        ez = R_["Z"][c] - y; ok = np.isfinite(ez) & np.isfinite(ec); ezs = pd.Series(np.where(ok, ez, np.nan)); ecs = pd.Series(np.where(ok, ec, np.nan))
        lz = ezs.groupby(R_["key"]).transform("mean").values; lc = ecs.groupby(R_["key"]).transform("mean").values
        print(f"  {L:4s} {c:5s} RMSE {rm(ez[ok]):.3f} (cell {rm(ec[ok]):.3f}) | corr all {np.corrcoef(ez[ok], ec[ok])[0, 1]:.2f} shape {np.corrcoef((ezs.values - lz)[ok], (ecs.values - lc)[ok])[0, 1]:.2f} level {np.corrcoef(lz[ok], lc[ok])[0, 1]:.2f}")
print("\n== leave-one-layout-out blend weight (w picked on other 3 layouts by mean relative MSE of assembled output) ==")
SUMMARY = {}
for c in CANDS:
    rel = {L: {w: (np.mean([mse(p - R_["y"]) / mse(b - R_["y"]) for p, b in zip(R_["P"][(c, w)], R_["base"])]) - 1) if w > 0 else 0.0 for w in WG} for L, R_ in RES.items()}
    out = {}
    for L in RES:
        wsel = min(WG, key=lambda w: (np.mean([rel[o][w] for o in RES if o != L]), w))
        out[L] = (wsel, 100 * (np.sqrt(1 + rel[L][wsel]) - 1))
    SUMMARY[c] = out
    print(f"  {c:5s} " + " | ".join(f"{L}: w {w:.2f} RMSE {d:+.2f}%" for L, (w, d) in out.items()) + "  || in-sample per-layout best w: " + " ".join(f"{L} {min(WG, key=lambda w: rel[L][w]):.2f}" for L in RES))
    for L, R_ in RES.items():
        w = out[L][0]
        if w > 0:
            ul = ~R_["linked"]; ch_ul = np.mean([rm((p - R_["y"])[ul]) / rm((b - R_["y"])[ul]) - 1 for p, b in zip(R_["P"][(c, w)], R_["base"])])
            t8 = np.nan_to_num(R_["t"], nan=99) <= 8; ch8 = np.mean([rm((p - R_["y"])[t8]) / rm((b - R_["y"])[t8]) - 1 for p, b in zip(R_["P"][(c, w)], R_["base"])]) if t8.any() else np.nan
            print(f"        {L} unlinked records {100 * ch_ul:+.2f}% | <=8C rows {100 * ch8:+.2f}%")
print("\nscreen rule (pre-registered): candidate goes to MEM2 judge only if LOLO w > 0 and RMSE change < 0 in all 4 layouts")
for c, out in SUMMARY.items(): print(f"  {c}: {'ADVANCE' if all(w > 0 and d < 0 for w, d in out.values()) else 'stop'}")
