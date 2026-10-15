# -*- coding: utf-8 -*-
"""ORT1 (2026-10-16 집 클로드; 사용자 '적어도 0.42 에는 들어가고 싶어' = 리더보드 .4706 -> .42, RMSE 약 -11%).  서술(오라클, 판정 없음), 실행 전 커밋.  온도.
질문: 어떤 정보를 알면 -11% 에 닿는가.  EXT10·EXT12·DIAG10 (TV3 틀), 기준 = R27 + CH5 (FL5 거울 연결).
오라클(정답 이어짐 표 linker_v3 truth.pkl cert, 진단 전용 - 제출에 쓰지 않음):
  O_link        모든 기록을 '진짜 전날'에 연결(전날 정답 있음 -> K8 앵커, 전날이 같은 폴드 앞번호 평가 기록 -> 시간 감쇠 사슬) = 링커가 완벽할 때 규정 안 최대
  O_linkK8only  위에서 앵커만(사슬 없음)          O_prevtrue  진짜 전날의 23시 배지 정답을 모든 연결 기록에서 안다고 가정(사슬 예측이 완벽할 때)
  O_0h          자기 0시 오차를 알고 exp(-h/4.5) 감쇠          O_day  자기 하루 수준 오차를 앎(상한)
출력: 배치별 진짜 전날 도달 범위(정답 있음/앞번호 평가/뒷번호 평가/없음), 현재 연결 중 진짜 비율, 오라클별 RMSE 변화와 리더보드 대략 환산(.4706 x (1 + 변화)).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ort1_temp_link_oracle_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "ort1_temp_link_oracle_v1.py")
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
TT = pd.read_pickle(os.path.join(env.LOCAL, "linker_v3", "truth.pkl")); TT = TT[TT.cert]
TRUEP = {f"{fb}_{int(nb):03d}": f"{fa}_{int(na):03d}" for fb, nb, fa, na in zip(TT.fb, TT.nb, TT.fa, TT.na)}
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3ort1"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
MIR5 = pd.read_csv(os.path.join(env.LOCAL, "fl5_mirror_links.csv")); OUT = []
for L in ("EXT10", "EXT12", "DIAG10"):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); M = M[np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TXo = tx3(M, key, .25, 10, .8); FO = dict(zip(key, M.fold.values)); HELD = {}
    for r, k in FO.items(): HELD.setdefault(k, set()).add(r)
    LK = {r: (p, q) for r, p, q in zip(MIR5[MIR5.layout == L].rec, MIR5[MIR5.layout == L].prev, MIR5[MIR5.layout == L].p)}; order = sorted(recs, key=lambda r: (r[:3], int(r[4:7])))
    AF, CH = link_plan(LK, recs, pos, h, FO, lambda r, p: p not in HELD[FO[r]])
    # oracle link plans (truth table, diagnosis only)
    cov = dict(none=0, lab=0, held_earlier=0, held_later=0, other=0); AFt, CHt, AFall = {}, {}, {}
    for r in recs:
        p = TRUEP.get(r)
        if p is None or p not in IDX or not np.isfinite(ST[IDX[p], 23]): cov["none"] += 1; continue
        AFall[r] = float(ST[IDX[p], 23])
        if p not in HELD[FO[r]]: cov["lab"] += 1; AFt[r] = float(ST[IDX[p], 23])
        elif p in pos and p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) and (h[pos[p]] == 23).any(): cov["held_earlier"] += 1; CHt[r] = p
        elif p in pos and p[:3] == r[:3]: cov["held_later"] += 1
        else: cov["other"] += 1
    cur = dict(K8=len(AF), chain=len(CH)); right_af = sum(TRUEP.get(r) == LK[r][0] for r in AF if r in TRUEP); right_ch = sum(TRUEP.get(r) == p for r, p in CH.items() if r in TRUEP)
    P = {k: [] for k in ("B", "O_link", "O_linkK8only", "O_prevtrue", "O_0h", "O_day")}
    for b in bs:
        for fam in "AB":
            x = B2.w40(M, b, cx, fam); base = assemble(x, AF, CH, pos, h, TXo, order); P["B"].append(base)
            P["O_link"].append(assemble(x, AFt, CHt, pos, h, TXo, order))                      # every record linked to its TRUE previous day (legal reach: labelled or earlier-numbered held)
            P["O_linkK8only"].append(assemble(x, AFt, {}, pos, h, TXo, order))                # true links, labelled anchors only (no chain)
            P["O_prevtrue"].append(assemble(x, AFall, {}, pos, h, TXo, order))               # true previous day AND its true 23h substrate known for every linked record
            o0 = base.copy()
            for r in recs:
                ix = pos[r]; e0 = (base - y)[ix][h[ix] == 0]
                if len(e0) and np.isfinite(e0[0]): o0[ix] -= np.exp(-h[ix] / 4.5) * e0[0]
            P["O_0h"].append(o0)                                                                  # own 0h error known, decaying
            P["O_day"].append(base - pd.Series(base - y).groupby(key).transform("mean").values)  # own day-level offset known
    for k, v in P.items():
        if k == "B": continue
        OUT.append(dict(layout=L, arm=k, rmse_base=np.mean([rm(q - y) for q in P["B"]]), rmse=np.mean([rm(q - y) for q in v]), chg=np.mean([rm(q - y) / rm(b - y) - 1 for q, b in zip(v, P["B"])])))
    print(f"{L}: records {len(recs)} | current plan K8 {cur['K8']} (true {right_af}) chain {cur['chain']} (true {right_ch}) | true-prev coverage {cov}", flush=True)
D = pd.DataFrame(OUT); pd.set_option("display.width", 200)
print(D.round(4).to_string(index=False))
S = D.groupby("arm").chg.mean()
print("\nmean change over 3 layouts and LB translation (.4706 x (1 + change); rough, eval composition differs):")
for k, c in S.items(): print(f"  {k:13s} {100 * c:+.1f}%  -> ~{.4706 * (1 + c):.3f}")
