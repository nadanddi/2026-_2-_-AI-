# -*- coding: utf-8 -*-
"""Environment bootstrap for independent analysts. `import boot` first.

Adds the bundled package folder to sys.path and registers the .libs DLL
folders so lightgbm loads on this machine (MSVC runtime is missing).
Exposes DATA (competition CSV folder) and PDF (problem statement).
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, ".analysis-tools", "python")
DATA = os.path.join(ROOT, "온라인대회자료", "정형데이터", "참가자_배포")
PDF = os.path.join(ROOT, "온라인대회자료", "정형데이터", "정형데이터_문제설명서.pdf")

if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
if hasattr(os, "add_dll_directory"):
    for d in sorted(set(glob.glob(os.path.join(TOOLS, "*", ".libs")) + glob.glob(os.path.join(TOOLS, "*.libs")))):
        try:
            os.add_dll_directory(d)
        except OSError:
            pass
