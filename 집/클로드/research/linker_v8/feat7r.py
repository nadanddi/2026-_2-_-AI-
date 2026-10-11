# feat7 with RELAXED hour-0 date matching (2026-10-12 집 클로드): at decision hours h<6 a same-date match needs >= RTHR of the
# outdoor cells of hours 0..h to be equal (v7: all cells).  Why (fragments #2-#4): same-date record pairs differ only in ONE
# hour-0 (or hour-23) outdoor cell by ~1 unit (midnight aggregation jitter), which blocks the exact hour-0 date match for some
# evaluation records (F13_185/201/206/228, F47_234).  Only the target's own hour-0..h inputs are used (causal).
import sys, re
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v7")
src=open(r"C:\work\farmai\집\클로드\research\linker_v7\feat7.py",encoding="utf-8").read()
src=src.replace("thr=1.0 if h<6 else 0.9","thr=RTHR if h<6 else 0.9")
RTHR=0.75
exec(compile(src,"feat7r","exec"))
