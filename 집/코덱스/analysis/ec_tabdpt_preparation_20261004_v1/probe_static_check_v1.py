import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

base = Path(r"C:\work\farmai\집\코덱스\analysis\ec_tabdpt_preparation_20261004_v1")
probe = base / "runtime_probe_draft_v1.py"
ast.parse(probe.read_text(encoding="utf-8"), filename=str(probe))
before = set(sys.modules)
runpy.run_path(str(probe), run_name="static_probe_import_only")
new_heavy = sorted((set(sys.modules) - before) &
                   {"torch", "numpy", "faiss", "tabdpt", "omegaconf",
                    "huggingface_hub", "safetensors"})
assert not new_heavy
print(json.dumps({"status": "PASS_STATIC_PROBE_ONLY",
                  "syntax": "PASS",
                  "heavy_imports": new_heavy,
                  "probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
                  "family_registration": None, "weight_reads": 0,
                  "actual_import_probe": 0, "fits": 0, "predictions": 0},
                 ensure_ascii=False, indent=2))

