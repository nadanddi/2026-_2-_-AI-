# TR81 (2026-10-12 집 클로드; 사용자: '나머지 81 은 어떻게 나뉘나') 서술: 학습 400 중 진단용 전날 연결이 달력상 하루 전이 아닌 81 기록 분해.
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); D = dict(zip(CAL.rec, CAL.day)); SPL = dict(zip(CAL.rec, CAL.split))
byday = CAL.dropna(subset=["day"]).groupby("day").rec.apply(list).to_dict()
TL = pd.read_pickle(S3 + r"\truth.pkl"); rows = []
for a, b, c in zip(TL.a1, TL.b, TL.cert):
    a, b = rid(int(a)), rid(int(b))
    if SPL.get(b) != "train": continue
    db, da = D.get(b, np.nan), D.get(a, np.nan); gap = db - da
    if gap == 1: continue
    prevday = byday.get(db - 1, []) if np.isfinite(db) else []
    labp = [x for x in prevday if SPL[x] == "train"]; evp = [x for x in prevday if SPL[x] == "eval"]
    # missing (deleted) numbers next to b in same ID
    f, n = b[:3], int(b[4:7]); deleted_nb = [f"{f}_{n - k:03d}" for k in (1, 2) if f"{f}_{n - k:03d}" not in SPL]
    if not np.isfinite(db) or not np.isfinite(da): cat = "A date unknown (no calendar day)"
    elif gap == 2: cat = "B skips one day (gap 2)"
    elif len(prevday) == 0: cat = "C no record at all on d-1 (calendar hole)"
    elif len(labp) == 0: cat = "D d-1 has only evaluation records"
    else: cat = "E d-1 has training records but link points elsewhere"
    rows.append(dict(rec=b, link=a, cert=bool(c), gap=gap, cat=cat, n_lab_prev=len(labp), n_eval_prev=len(evp), p2=n >= 179, deleted_nb=",".join(deleted_nb)))
T = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(f"81-check: {len(T)}"); print(pd.crosstab(T.cat, T.cert, margins=True).to_string())
print("\nE detail (link elsewhere although training records exist on d-1):"); print(T[T.cat.str.startswith("E")].sort_values("gap")[["rec", "link", "cert", "gap", "n_lab_prev", "n_eval_prev", "p2"]].to_string(index=False))
print("\nB/C/D detail:"); print(T[T.cat.str[0].isin(list("BCD"))][["rec", "link", "cert", "gap", "cat", "n_lab_prev", "n_eval_prev", "deleted_nb"]].to_string(index=False))
