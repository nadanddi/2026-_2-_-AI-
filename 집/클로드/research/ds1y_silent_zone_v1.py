# -*- coding: utf-8 -*-
"""DS1Y (2026-10-13 집 클로드; 사용자: 'HMM 손해·이득 기록이 침묵 구간에 얼마나 있는지').  서술(판정 아님).  EC, DIAG10.
DS1X(6.816) 기록별 ΔSSE 를 세 부품의 발동 여부로 나눔(각 기록의 DIAG10 시험 폴드 기준 — 그 폴드 숨김 기록의 정답은 안 보임):
  EA2 = v7 확신 짝(p ≥ .8·정답 짝·dT ≤ 1.5)의 짝이 그 폴드에서 숨김이 아님;  TNA8 = v15R 거울 짝이 정답 기록이고 (p0 ≥ .8 또는 HP2 일치), 숨김 아님;
  B5 = 어떤 시각 h 에서 sg 0..h 누적평균 ≥ 1.0 & 같은 날 학습 쌍둥이(0..h 입력, 숨김 아님) 중 하루 평균 ≥ 1.2 (3시드 중 하나라도).
침묵 = 셋 다 아님.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
CR1SRC = open(os.path.join(HERE, "cr1_chain_gru_temp_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "cr1_chain_gru_temp_v1.py")
exec(CR1SRC[:CR1SRC.index("V = [\"in_temp\", \"in_hum\"")])   # v7set, L7, PM, TNA8, idx, TRAIN, FARM, NUM, N
__file__ = os.path.join(HERE, "ds1y_silent_zone_v1.py")
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X  # noqa: E402
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
XR = np.round(X, 1)
def zflag(i, h, vis):
    c = np.where(vis)[0]; c = c[c != i]; t = c[np.all(np.abs(XR[c, :h + 1] - XR[i, :h + 1]) < 1e-6, axis=(1, 2))]; t = t[np.isfinite(DM[t])]
    return bool((DM[t] >= 1.2).any())
V7P = {r: p for r, p in zip(L7.rec, L7.prev) if r in v7set}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
fold = O.groupby("rec").fold.first().to_dict(); hidden = O.groupby("fold").rec.apply(set).to_dict()
D = pd.read_csv(os.path.join(HERE, "ds1x_case_rows_v1.csv"))
b5 = {}
for sd, G in O.groupby("seed"):
    G = G.sort_values(["rec", "hour"]); G["cm"] = G.groupby("rec").sg.transform(lambda s: s.expanding().mean())
    for r, g in G.groupby("rec"):
        if b5.get(r): continue
        hs = hidden[fold[r]]; vis = TRAIN & ~np.array([rid(k) in hs for k in range(N)])
        b5[r] = any(zflag(idx[r], int(h), vis) for h, c in zip(g.hour, g.cm) if c >= 1.0)
D["EA2"] = [r in V7P and V7P[r] not in hidden[fold[r]] for r in D.rec]
D["TNA8"] = [r in TNA8 and TNA8[r] not in hidden[fold[r]] for r in D.rec]
D["B5"] = D.rec.map(b5).fillna(False)
D["zone"] = np.select([D.EA2 | D.TNA8, D.B5], ["anchor", "B5"], "silent")
D["predH"] = D.pred >= 1.0
D["type"] = np.select([D.trueH & D.predH, D.trueH & ~D.predH, ~D.trueH & D.predH], ["H_seen", "H_missed", "N_falseH"], "N_ok")
pd.set_option("display.width", 220)
print("zone sizes:", D.zone.value_counts().to_dict(), "| EA2", int(D.EA2.sum()), "TNA8", int(D.TNA8.sum()), "B5", int(D.B5.sum()))
print("\n[zone x type] n / sum dSSE")
print(D.pivot_table(index="type", columns="zone", values="dsse", aggfunc=["size", "sum"]).round(2).to_string())
print("\n[zone x pass] sum dSSE"); print(D.pivot_table(index="p2", columns="zone", values="dsse", aggfunc="sum").round(2).to_string())
print("\nsilent-only total dSSE: all", round(D[D.zone == "silent"].dsse.sum(), 2), "| pass-2", round(D[(D.zone == "silent") & D.p2].dsse.sum(), 2))
cols = ["rec", "type", "zone", "EA2", "TNA8", "B5", "y", "pred", "dsse"]
print("\n[top 10 gains]"); print(D.sort_values("dsse").head(10)[cols].round(3).to_string(index=False))
print("\n[top 10 losses]"); print(D.sort_values("dsse").tail(10)[cols].round(3).to_string(index=False))
D.to_csv(os.path.join(HERE, "ds1y_zone_rows_v1.csv"), index=False)
