import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, glob, os, csv
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8")
L = r"G:\내 드라이브\farmai_sync\lab_claude_local\fp_db"
M = r"C:\work\farmai\제출\최종검증\팜모니_이미지\data\manifests\server"
cnt = Counter(); grp = defaultdict(set); tarf = defaultdict(set); var = defaultdict(set)
for p in sorted(glob.glob(L + r"\*.npz")):
    t = os.path.basename(p)[:-4]
    for n in np.load(p, allow_pickle=True)["names"]:
        f = str(n).split("_")[0]; cnt[f] += 1; grp[f].add("_".join(str(n).split("_")[:3])); tarf[f].add(t)
        var[f].add(t.split(".")[1].rstrip("0123456789") + ("(V)" if t.startswith("VS") else ""))
TR = ["im_set_v4","im_set_sd2","im_set_v5","im_set_gpt1","im_set_gpt2","im_set_newreal","im_set_v7tr","im_set_qwentr","im_set_h3","im_set_h3j","im_set_gptr","im_set_gptr_src","im_set_gpt3"]
EV = ["im_set_r2","im_set_r3","im_set_r4","im_set_r5"]
def farms(sets):
    s = set()
    for e in sets:
        for r in csv.DictReader(open(os.path.join(M, e, "manifest.csv"), encoding="utf-8")):
            if r.get("farm_code"): s.add(r["farm_code"])
    return s
trf, evf = farms(TR), farms(EV)
print("AI Hub 전체 사진", sum(cnt.values()), "농장", len(cnt))
print("학습 농장", len(trf), "봉인 농장", len(evf), "봉인 중 학습 밖", len(evf - trf))
un = sorted(set(cnt) - trf)
print("학습에 안 쓴 농장", len(un), "사진", sum(cnt[f] for f in un))
for f in un: print(f, cnt[f], "묶음", len(grp[f]), "품종", sorted(var[f]), "tar", len(tarf[f]))
print("--- 학습 농장 사진 합", sum(cnt[f] for f in trf))
print("품종별 농장(전체):", Counter(min(var[f]) for f in cnt))
print("품종별 농장(미사용):", Counter(min(var[f]) for f in un))
