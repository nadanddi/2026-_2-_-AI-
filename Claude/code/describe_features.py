# -*- coding: utf-8 -*-
"""Group the submitted model's features by kind, for explanation."""
import collections
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def group(c):
    if re.search(r"_ewm\d+$", c):
        return "지수가중평균(EWM) - 최근일수록 크게 섞은 평균"
    if "_dev" in c:
        return "현재값 - 평균 (편차)"
    if re.search(r"_lag\d+$", c):
        return "n시간 전 값 (lag)"
    if c.endswith(("_d1", "_d3")):
        return "변화량 (차분)"
    if re.search(r"_r\d+m$|_r\d+s$", c):
        return "롤링 평균/표준편차 (24·72·168시간)"
    if "_pd7m" in c:
        return "직전 7일 평균"
    if re.search(r"_pd[123]$", c):
        return "전일/전전일 하루 집계"
    if c.endswith(("_tdmean", "_tdsum")):
        return "당일 0시~현재 누적"
    if c.endswith(("_cum", "_cumrate")):
        return "작기 전체 누적"
    if c.startswith("rtr_") or c.startswith("transp_per_rad"):
        return "RTR 등 비율 지표"
    if c in ("day", "hour", "hr_sin", "hr_cos", "farm_id", "day_par", "midnight"):
        return "시간 / 온실 정보"
    return "현재 시각의 원본·파생값"


m = json.load(open(os.path.join(ROOT, "models_ec_current", "manifest.json"), encoding="utf-8"))
for t in ("sub_temp", "sub_ec"):
    cols = m["features"][t]
    cnt = collections.Counter(group(c) for c in cols)
    print("=== %s : 총 %d개 ===" % (t, len(cols)))
    for k, v in cnt.most_common():
        ex = [c for c in cols if group(c) == k][:3]
        print("  %3d개  %-40s 예: %s" % (v, k, ", ".join(ex)))
    print()
