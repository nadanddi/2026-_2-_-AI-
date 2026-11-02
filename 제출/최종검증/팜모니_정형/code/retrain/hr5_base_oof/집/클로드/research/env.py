# -*- coding: utf-8 -*-
"""Reproduction-tree bootstrap (replaces the research env.py of the original repository).

The original research modules expect the repository layout (집/클로드/research, 집/코덱스/analysis, 연구실/클로드/code).
This folder keeps that layout; ROOT is the folder that contains 집/ and 연구실/.

  AGRI_DATA   folder with the ORIGINAL competition train_X.csv / train_y.csv / test_X.csv (required)
  HR5_LOCAL   scratch folder for intermediate files (default: <ROOT>/local)

Packages come from the normal Python environment (code/requirements.txt).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # <ROOT>/집/클로드/research
CODE = os.path.join(ROOT, u"연구실", u"클로드", "code")
DATA = os.environ.get("AGRI_DATA") or os.path.join(ROOT, "data")
LOCAL = os.environ.get("HR5_LOCAL") or os.path.join(ROOT, "local")
CODEX = os.path.join(ROOT, u"집", u"코덱스", "analysis", "codex_independent")
CODEX_PATHS = [CODEX]
TOOLS = os.path.join(ROOT, ".analysis-tools", "python")   # not used; kept because env_extra reads it


def bootstrap():
    if CODE not in sys.path:
        sys.path.insert(0, CODE)
    for p in CODEX_PATHS:
        if p not in sys.path:
            sys.path.append(p)
    os.makedirs(LOCAL, exist_ok=True)
    import common
    common.DATA = DATA
    return common


common = bootstrap()
