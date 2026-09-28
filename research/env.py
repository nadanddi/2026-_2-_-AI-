# -*- coding: utf-8 -*-
"""Bootstrap: DLL search path, module path, and the real data location.

Why this file exists
--------------------
1. The Visual C++ / OpenMP runtimes are NOT installed system-wide on this
   machine, so lightgbm's lib_lightgbm.dll fails to load.  numpy, scipy and
   sklearn each ship their own copies under a `.libs` folder; registering
   those folders as DLL directories makes lightgbm load.  Import order does
   not matter once the directories are registered.
2. Claude/code/common.py resolves DATA to Claude/정형데이터, which does not
   exist.  The real CSVs live under 온라인대회자료/.  We patch the module
   global instead of copying 21 MB of competition data around.

Usage:  import env   # must be the first project import
"""
import os
import glob
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, ".analysis-tools", "python")
CODE = os.path.join(ROOT, "Claude", "code")
DATA = os.path.join(ROOT, u"온라인대회자료", u"정형데이터", u"참가자_배포")
LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local")


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
    _register_dll_dirs()
    os.makedirs(LOCAL, exist_ok=True)
    import common
    common.DATA = DATA
    return common


common = bootstrap()
