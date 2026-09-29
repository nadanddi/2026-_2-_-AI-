# -*- coding: utf-8 -*-
"""Bootstrap: DLL search path, module path, and the real data location.

Why this file exists
--------------------
1. The Visual C++ / OpenMP runtimes are NOT installed system-wide on this
   machine, so lightgbm's lib_lightgbm.dll fails to load.  numpy, scipy and
   sklearn each ship their own copies under a `.libs` folder; registering
   those folders as DLL directories makes lightgbm load.  Import order does
   not matter once the directories are registered.
2. 연구실/클로드/code/common.py resolves DATA to 연구실/클로드/정형데이터, which
   does not exist.  The real CSVs live under 공용/대회자료/.  We patch the
   module global instead of copying 21 MB of competition data around.
3. 2026-09-29 folder reorganisation (집/연구실 x 클로드/코덱스, 공용, 제출):
   ROOT is now found by walking up to CLAUDE.md, Codex modules that old
   scripts reached through "../analysis/..." are put on sys.path here, and
   submission files moved from research/submissions to 제출/ (see
   submission_path()).

Usage:  import env   # must be the first project import
"""
import os
import glob
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_root(start):
    d = start
    while True:
        if os.path.isfile(os.path.join(d, "CLAUDE.md")) and os.path.isdir(os.path.join(d, u"공용")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("repository root (CLAUDE.md + 공용/) not found above " + start)
        d = parent


ROOT = _find_root(HERE)
TOOLS = os.path.join(ROOT, ".analysis-tools", "python")
CODE = os.path.join(ROOT, u"연구실", u"클로드", "code")
DATA = os.path.join(ROOT, u"공용", u"대회자료", u"정형데이터", u"참가자_배포")
LOCAL = os.path.join(HERE, "local")
SUBMIT = os.path.join(ROOT, u"제출")
CODEX = os.path.join(ROOT, u"집", u"코덱스", "analysis", "codex_independent")
CODEX_PATHS = [CODEX, os.path.join(CODEX, u"2차")]


def submission_path(name):
    """Find a file that used to live in research/submissions (e.g. 'submission_06.csv')."""
    hits = glob.glob(os.path.join(SUBMIT, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(name + " not found under " + SUBMIT)
    return hits[0]


def _register_dll_dirs():
    if not hasattr(os, "add_dll_directory"):
        return []
    found = sorted(set(glob.glob(os.path.join(TOOLS, "*", ".libs"))
                       + glob.glob(os.path.join(TOOLS, "*.libs"))))
    for d in found:
        try:
            os.add_dll_directory(d)
        except OSError:
            pass
    return found


def bootstrap():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    if CODE not in sys.path:
        sys.path.insert(0, CODE)
    for p in CODEX_PATHS:
        if p not in sys.path:
            sys.path.append(p)
    _register_dll_dirs()
    os.makedirs(LOCAL, exist_ok=True)
    import common
    common.DATA = DATA
    return common


common = bootstrap()
