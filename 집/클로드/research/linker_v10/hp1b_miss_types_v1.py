# HP1b (2026-10-12 집 클로드) 서술: 확실 연결 이을 수 있는 65 에서 v10(0시) 1순위가 틀린 기록 — 틀린 선택이 '같은 날짜 다른 동'인지 '다른 날짜'인지,
# 같은 날짜면 0..h 지문으로 그 날짜 안에서 다시 고를 때 맞는지(h 0/9/23).
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
exec(open("hp1_hour_progressive_match_v1.py", encoding="utf-8").read().split("TRG = train_groups10()")[0])
CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); DAYr = dict(zip(CAL.rec, CAL.day)); DAY = np.array([DAYr.get(rid(i), np.nan) for i in range(N)])
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
            if not TRAIN[b] or b not in TP or not CERT[b] or not lab[TP[b]]: continue
            D, _ = features10(b, 0, lab, removed=rem2)
            a = D.a.values.astype(int); sc = D[FEATS10].values.astype(float) @ w; p = np.exp(sc - sc.max()); p /= p.sum(); k = int(np.argmax(sc)); pick = a[k]; t = TP[b]
            if pick == t: continue
            same = DAY[pick] == DAY[t]
            row = dict(rec=rid(b), pick=rid(pick), truth=rid(t), p=round(float(p[k]), 3), pick_lab=bool(lab[pick]), same_date=bool(same), d_pick=DAY[pick], d_truth=DAY[t], d_rec=DAY[b])
            if same:
                sib = [x for x in a if DAY[x] == DAY[t] and lab[x]]
                for h in (0, 9, 23):
                    d = {x: np.nanmean(np.minimum(np.abs(FPA[x, :h + 1] - FPA[b, :h + 1]), 4)) for x in sib}
                    row[f"fp_ok_h{h}"] = min(d, key=d.get) == t
            rows.append(row)
T = pd.DataFrame(rows); pd.set_option("display.width", 220); print(T.to_string(index=False))
