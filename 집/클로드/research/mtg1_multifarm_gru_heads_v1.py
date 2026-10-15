# -*- coding: utf-8 -*-
"""MTG1 (2026-10-16 집 클로드; 사용자 '둘 다 진행해봐' — 51개 온실 동시 학습 RNN + 온실별 머리(MTL), 이전 CR3(사전학습 후 미세조정)·H-P1(범주)·TT20(낮은 가중)과 다른 형태).
사전 고정(커밋 후 실행).  온도.
모델: 하루(0..23시) 인과 GRU 2층 64 + 온실 임베딩 8(입력에 붙임) + 머리 = 공용 선형 + 온실별 편차 선형(0 초기화) + 온실별 절편.
  입력: 공통 in_temp·in_hum·in_co2(표준화 + 있음 표시), F13·F47 전용 out_temp·out_rad·act_heating·act_thermal·act_vent(다른 온실은 0 + 없음 표시), 시각 sin·cos.  하루 안만(이어붙임 규칙과 같음).
  정제: LOGO1 과 같음.  손실: 가중 MSE, F13·F47 행 합 = 다른 온실 행 합(절반씩), F13·F47 행은 WB(거친 날) 가중.  Adam 2e-3, 20 에폭, 배치 512, 시드 3·4 평균.
  폴드: 각 배치(CV8·N·CK2·EL1) 폴드마다 학습 = 다른 49개 온실 전체 + 그 폴드 F13·F47 학습 기록, 예측 = 검증 기록 (local/mtg1_ckpt).
섞기(기준 = 셀 -> V3LF 30% 8/10 -> FL K8 -> TX3; MTL 은 GRU30 다음·앵커 전):  ALL = .2 * (MTL - q) 모든 행 | COLD = .3 * clip((10 - in_temp)/2) * (MTL - q).
판정(본페로니 2 -> alpha .0125): CV8·N·CK2 12셀 모두 개선 & 합친 5일 블록 P(악화) < .0125 & EL1 안전.  서술: MTL 단독 vs 셀 온도 구간별 RMSE·편향·오차 상관.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u mtg1_multifarm_gru_heads_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1(OS1 v2), B, BETA, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "mtg1_multifarm_gru_heads_v1.py")
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
import env_extra_gpu  # noqa: F401
import torch, torch.nn as nn
CKM = os.path.join(env.LOCAL, "mtg1_ckpt"); os.makedirs(CKM, exist_ok=True)
XA = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); YA = pd.read_csv(os.path.join(env.DATA, "train_y.csv"), usecols=["row_id", "sub_temp"])
A = XA.merge(YA, on="row_id", how="left"); A["farm"] = A.row_id.str[:3]; A["day"] = A.row_id.str[4:7].astype(int); A["hour"] = A.row_id.str[8:10].astype(int)
A.loc[(A.in_temp < 0) | (A.in_temp > 45), "in_temp"] = np.nan; A.loc[A.in_hum <= 0, "in_hum"] = np.nan; A.loc[A.in_co2 <= 0, "in_co2"] = np.nan; A.loc[A.sub_temp >= 40, "sub_temp"] = np.nan
COM = ["in_temp", "in_hum", "in_co2"]; EXT = ["out_temp", "out_rad", "act_heating", "act_thermal", "act_vent"]
cols = []
for c in COM + EXT:
    mu, sd = A[c].mean(), A[c].std() or 1.0; A["z_" + c] = ((A[c] - mu) / sd).fillna(0.0); A["k_" + c] = A[c].notna().astype(float); cols += ["z_" + c, "k_" + c]
A["hs"], A["hc"] = np.sin(2 * np.pi * A.hour / 24), np.cos(2 * np.pi * A.hour / 24); cols += ["hs", "hc"]
A["rec"] = A.farm + "_" + A.day.map("{:03d}".format); RECS = sorted(A.rec.unique()); RI = {r: i for i, r in enumerate(RECS)}; FARMS = sorted(A.farm.unique()); FI = {f: i for i, f in enumerate(FARMS)}
NR = len(RECS); XS = np.zeros((NR, 24, len(cols)), np.float32); YS = np.full((NR, 24), np.nan, np.float32)
ri = A.rec.map(RI).values; hh = A.hour.values; XS[ri, hh] = A[cols].to_numpy(np.float32); YS[ri, hh] = A.sub_temp.to_numpy(np.float32)
FREC = np.array([FI[r[:3]] for r in RECS]); TGT = np.isin(np.array([r[:3] for r in RECS]), ["F13", "F47"]); YM = 16.6
WROW = dict(zip(lab.row_id.values, WB))   # rough-day weights for F13/F47 rows (same as other members)
WS = np.ones((NR, 24), np.float32)
for rid, w in WROW.items():
    r = rid[:7]
    if r in RI: WS[RI[r], int(rid[8:10])] = w
class Net(nn.Module):
    def __init__(s, nf, din, emb=8, hid=64):
        super().__init__(); s.emb = nn.Embedding(nf, emb); s.gru = nn.GRU(din + emb, hid, 2, batch_first=True, dropout=.1)
        s.sw = nn.Linear(hid, 1); s.hw = nn.Embedding(nf, hid); s.hb = nn.Embedding(nf, 1); nn.init.zeros_(s.hw.weight); nn.init.zeros_(s.hb.weight)
    def forward(s, x, f):
        e = s.emb(f)[:, None, :].expand(-1, x.shape[1], -1); h, _ = s.gru(torch.cat([x, e], -1))
        return s.sw(h).squeeze(-1) + (h * s.hw(f)[:, None, :]).sum(-1) + s.hb(f)
DEV = "cuda"
def train_predict(tr_idx, va_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    y = YS[tr_idx] - YM; m = np.isfinite(y); w = WS[tr_idx] * m; tg = TGT[tr_idx]
    tot_t, tot_o = w[tg].sum(), w[~tg].sum(); w[tg] *= tot_o / max(tot_t, 1e-6)   # F13/F47 = half of the loss
    X = torch.tensor(XS[tr_idx], device=DEV); Y = torch.tensor(np.nan_to_num(y), device=DEV); Wt = torch.tensor(w, device=DEV); F = torch.tensor(FREC[tr_idx], device=DEV)
    net = Net(len(FARMS), XS.shape[2]).to(DEV); opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-5); n = len(tr_idx)
    for ep in range(20):
        perm = torch.randperm(n, device=DEV)
        for b in range(0, n, 512):
            ix = perm[b:b + 512]; p = net(X[ix], F[ix]); loss = (Wt[ix] * (p - Y[ix]) ** 2).sum() / Wt[ix].sum().clamp_min(1e-6)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval()
    with torch.no_grad(): out = net(torch.tensor(XS[va_idx], device=DEV), torch.tensor(FREC[va_idx], device=DEV)).cpu().numpy() + YM
    return out
OTHER = np.where(~TGT)[0]
LAYS = (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks), ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask))
for L, ckb, ckg, folds, mk in LAYS:
    for k, fd in enumerate(folds):
        path = os.path.join(CKM, "%s_%d.csv" % (L, k))
        if os.path.exists(path): continue
        tm, vm = mk(lab, fd)
        if not vm.sum(): continue
        trr = sorted(set(lab.row_id[tm].str[:7])); var = sorted(set(lab.row_id[vm].str[:7]))
        tr_idx = np.concatenate([OTHER, np.array([RI[r] for r in trr if r in RI])]); va_idx = np.array([RI[r] for r in var])
        P = np.mean([train_predict(tr_idx, va_idx, s) for s in (3, 4)], 0)
        vr = lab.row_id[vm].values; out = pd.DataFrame({"row_id": vr, "MTL": [P[var.index(r[:7]), int(r[8:10])] for r in vr]})
        out.to_csv(path, index=False); print(f"{L}/{k} mtl done | alone rmse {rm(out.MTL.values - lab.sub_temp[vm].values):.3f}", flush=True)
RES, TAB = {}, []
for L, ckb, ckg, folds, mk in LAYS:
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(ckg, f)) for f in sorted(os.listdir(ckg)) if f.startswith(L + "_")], ignore_index=True)
    Md = pd.concat([pd.read_csv(os.path.join(CKM, f)) for f in sorted(os.listdir(CKM)) if f.startswith(L + "_")], ignore_index=True).drop_duplicates("row_id")
    M = M.merge(Gd, on="row_id", how="left").merge(Md, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    v = M.V3LF.values; ok = ~np.isnan(v); vv = np.nan_to_num(v); TXo = tx3(M, key, .25, 10, .8); G8 = gate(M, 8, 10); t = M.in_temp.values
    mt = M.MTL.values; okm = ~np.isnan(mt); mtv = np.nan_to_num(mt); gc = np.nan_to_num(np.clip((10 - t) / 2, 0, 1), nan=0.0)
    TRS = {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs
    AF = {r: float(ST[IDX[FLK[r][0]], 23]) for r in recs if r in FLK and r in TRS and FLK[r][1] >= .8 and FLK[r][0] in TRS[r] and np.isfinite(ST[IDX[FLK[r][0]], 23])
          and abs(IN_[IDX[r], 0, 0] - IN_[IDX[FLK[r][0]], 23, 0]) <= 1.5}
    def apply(x, AF=AF, pos=pos, h=h):
        out = x.copy()
        for r, a in AF.items():
            ix = pos[r]; x0 = x[ix][h[ix] == 0]
            if len(x0): out[ix] += BETA[h[ix]] * np.clip(a - BIAS - x0[0], -2, 2)
        return out
    P = {"B": [], "ALL": [], "COLD": []}; Q = []
    for s in O1.SEEDS:
        for fam in "AB":
            q = B.w40(M, "base_R_%d" % s, "codex_R", fam); q = q + .30 * G8 * ok * (vv - q); Q.append(q)
            P["B"].append(apply(q) + TXo); P["ALL"].append(apply(q + .2 * okm * (mtv - q)) + TXo); P["COLD"].append(apply(q + .3 * gc * okm * (mtv - q)) + TXo)
    RES[L] = dict(P=P, y=y, t=t, bk=(L + "_" + M.farm + "_" + (M.day // 5).astype(str)).values)
    qm = np.mean(Q, 0)
    for nm, lo, hi in (("<=6", -99, 6), ("6-8", 6, 8), ("8-10", 8, 10), ("10-12", 10, 12), (">12", 12, 99)):
        m = (np.nan_to_num(t, nan=99) > lo) & (np.nan_to_num(t, nan=99) <= hi) & okm
        if m.sum() >= 10: TAB.append(dict(set=L, band=nm, rows=int(m.sum()), cells=rm((qm - y)[m]), MTL=rm((mt - y)[m]), b_cells=float((qm - y)[m].mean()), b_MTL=float((mt - y)[m].mean()), corr_err=float(np.corrcoef((qm - y)[m], (mt - y)[m])[0, 1])))
    print(L, "eval done", flush=True)
pd.set_option("display.width", 250); print("\n[desc] MTL alone vs cells(+GRU30) by band"); print(pd.DataFrame(TAB).round(3).to_string(index=False))
def cellch(L, c, m=None):
    R_ = RES[L]; y = R_["y"]; m = np.ones(len(y), bool) if m is None else m
    return [rm((p - y)[m]) / rm((b - y)[m]) - 1 for p, b in zip(R_["P"][c], R_["P"]["B"])]
rng = np.random.default_rng(20270150)
for c in ("ALL", "COLD"):
    for L in RES:
        tt = np.nan_to_num(RES[L]["t"], nan=99)
        print(f"  {c} {L}: cells {' '.join('%+.3f' % (100 * x) for x in cellch(L, c))} | <=6 {100 * np.mean(cellch(L, c, tt <= 6)) if (tt <= 6).sum() >= 5 else float('nan'):+.2f}% | 6-8 {100 * np.mean(cellch(L, c, (tt > 6) & (tt <= 8))):+.2f}% | >10 {100 * np.mean(cellch(L, c, tt > 10)):+.2f}%")
    cells = cellch("CV8", c) + cellch("N", c) + cellch("CK2", c)
    dd = pd.concat([pd.Series(np.mean([(p - RES[L]["y"]) ** 2 - (b - RES[L]["y"]) ** 2 for p, b in zip(RES[L]["P"][c], RES[L]["P"]["B"])], 0)).groupby(RES[L]["bk"]).mean() for L in ("CV8", "N", "CK2")]).values
    pw = float((dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1) >= 0).mean())
    R_ = RES["EL1"]; c2 = cellch("EL1", c); d2 = pd.Series(np.mean([(p - R_["y"]) ** 2 - (b - R_["y"]) ** 2 for p, b in zip(R_["P"][c], R_["P"]["B"])], 0)).groupby(R_["bk"]).mean().values
    pb = float((d2[rng.integers(0, len(d2), (20000, len(d2)))].mean(1) < 0).mean()); safe = not (np.mean(c2) > 0 and pb < .025)
    print(f"MTG1 {c}: 12 cells all better {all(x < 0 for x in cells)} | combined block P(worse) {pw:.4f} (alpha .0125) | EL1 {100 * np.mean(c2):+.3f}% safe {safe} -> {'PASS' if all(x < 0 for x in cells) and pw < .0125 and safe else 'FAIL'}")
