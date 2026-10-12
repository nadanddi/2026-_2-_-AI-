# HP3 (2026-10-12 집 클로드; 사용자: 'HP2 규칙으로 171 의 정확도·포괄률') 서술: 거울 CV 400 기록 전부에 HP2 규칙(0시)을 적용.
#  층 A = DF 날짜 있음(외기 정확 일치) → 그 날짜−1 정답 기록 중 0시 지문 최근접;  층 B = DF 날짜 없음 → v10 1순위 날짜의 정답 기록 중 지문 최근접.
#  앵커 = 고른 기록(정답 기록) & |dT_air| ≤ 1.5.  정확도 = 맞음 / 앵커(진실 알려진), 포괄률 = 맞음 / 이을 수 있는(전체 171, 확실 65).
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
exec(open("hp2_date_then_hour_fp_v1.py", encoding="utf-8").read().split(chr(10) + "TRG = train_groups10(); rows = []; HS")[0])
TRG = train_groups10(); rows = []
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist()); rem2 = rem | oth
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        w = fit_clogit([(X_, y_) for b_, a_, X_, y_ in TRG if not touch[b_] and not touch[a_]], l2=0.5, iters=300)
        lab = TRAIN.copy(); lab[touch] = False
        for b in blk:
            if not TRAIN[b]: continue
            D, _ = features10(b, 0, lab, removed=rem2)
            if len(D) == 0: continue
            a = D.a.values.astype(int); sc = D[FEATS10].values.astype(float) @ w; top1 = a[np.argmax(sc)]
            vi = np.where(lab)[0]; eq = np.all(np.abs(X[vi, 0] - X[b, 0]) < 1e-6, axis=1); ds = DAY[vi[eq]]; ds = ds[np.isfinite(ds)]
            if len(ds): dprev = pd.Series(ds).mode().iloc[0] - 1; tier = "A"
            else: dprev = DAY[top1]; tier = "B"
            cand = [x for x in vi if DAY[x] == dprev and not any(lab[s_] and DAY[s_] == dprev + 1 for s_ in SUCC.get(x, []))]
            if not cand: continue
            pick = min(cand, key=lambda x: np.nanmean(np.minimum(np.abs(FPA[x, 0] - FPA[b, 0]), 4)))
            if abs(IN_[b, 0, 0] - IN_[pick, 23, 0]) > 1.5: continue
            t = TP.get(b); p1 = np.exp(sc - sc.max()); p1 /= p1.sum(); rows.append(dict(rec=rid(b), tier=tier, pick=rid(pick), v10=rid(top1), v10p=float(p1.max()), known=t is not None, cert=CERT.get(b, False), ok=(pick == t), linkable=(t is not None and bool(lab[t]))))
T = pd.DataFrame(rows)
nl_all = 171; nl_cert = 65
for name, S in (("tier A only", T[T.tier == "A"]), ("A + B", T)):
    k = S[S.known]; kc = S[S.cert]
    print(f"{name:12s}: anchors {len(S)} | all-truth precision {k.ok.mean():.3f}, recall {k.ok.sum()/nl_all:.2f} | cert precision {kc.ok.mean():.3f}, recall {kc.ok.sum()/nl_cert:.2f} | anchors on non-linkable records {int((~S.linkable).sum())}")
print("reference: v10 (p>=.8) 76 anchors all .921/.41, cert 1.000/.58 | META2 (>=.7) 80 all .975/.46, cert 1.000/.68")

L2 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "lr2_candidates.csv")); B2 = L2[L2.lab].loc[lambda d: d.groupby("rec").pm.idxmax()].set_index("rec")
T["meta"] = T.rec.map(B2.cand); T["metap"] = T.rec.map(B2.pm).fillna(0)
for name, S in (("A & agrees v10 top-1", T[(T.tier == "A") & (T.pick == T.v10)]), ("A+B & agrees v10 top-1", T[T.pick == T.v10]),
                ("A & agrees META", T[(T.tier == "A") & (T.pick == T.meta)]), ("A+B & agrees META", T[T.pick == T.meta]),
                ("A+B & agrees META & metap>=.5", T[(T.pick == T.meta) & (T.metap >= .5)])):
    k = S[S.known]; kc = S[S.cert]
    print(f"{name:30s}: anchors {len(S)} | all-truth precision {k.ok.mean():.3f}, recall {k.ok.sum()/nl_all:.2f} | cert precision {kc.ok.mean():.3f}, recall {kc.ok.sum()/nl_cert:.2f} | on non-linkable {int((~S.linkable).sum())}")
