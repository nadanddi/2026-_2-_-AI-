"""준비 코드의 정적/차단 관문만 검사한다. 데이터·가중치·후보 package import 없음."""
from pathlib import Path
import ast
import hashlib
import json
import runpy
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
candidate = HERE / "adapter_draft_v1.py"
syntax = {}
for file in (candidate, HERE/"inventory_readonly_v1.py", Path(__file__)):
    ast.parse(file.read_text(encoding="utf-8"))
    syntax[file.name] = "PASS"
before = set(sys.modules)
ns = runpy.run_path(str(candidate), run_name="static_import_only")
blocked = {}
for name, args in [("verify_weight", (Path("DO_NOT_READ.safetensors"),)),
                   ("activate_runtime", (Path("DO_NOT_READ"),)),
                   ("verify_package_pin", ()),
                   ("build_and_fit", (None, None, Path("DO_NOT_READ"), 7)),
                   ("predict_mean", (None, None, 7)),
                   ("raw_query_audit", (None, None, 7)),
                   ("combine_candidate", (None, None, None, None, 0., 1.))]:
    try:
        ns[name](*args)
    except RuntimeError as error:
        assert "준비 초안" in str(error)
        blocked[name] = "PASS"
    else:
        raise AssertionError(name + " runtime gate did not block")
heavy = {"torch", "numpy", "tabdpt", "faiss", "omegaconf", "huggingface_hub"}
unexpected = sorted(heavy.intersection(set(sys.modules)-before))
assert not unexpected, unexpected
core = ROOT/"집/코덱스/analysis/codex_independent/rl_ec_v1/run.py"
tree = ast.parse(core.read_text(encoding="utf-8"))
names = {"ACTS","INDOOR","RAW","BASE","FP","FULL"}
nodes = []
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names:
        nodes.append(node)
    if isinstance(node, ast.AugAssign) and isinstance(node.target,ast.Name) and node.target.id in names:
        nodes.append(node)
constants = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "<core_constants_only>", "exec"), constants)
expected = tuple(c for c in constants["FULL"] if c!="day")+("season",)
assert ns["FULL38"] == expected and len(expected)==38 and len(set(expected))==38
sources = [core, ROOT/"집/코덱스/analysis/statistical_experiments_20261003_v1/support.py",
           ROOT/"집/코덱스/analysis/ec_log_partition_mean_20261004_v1/run.py"]
result = dict(status="PASS_STATIC_ONLY_RUNTIME_UNTESTED", syntax=syntax,
              blocked_runtime_calls=blocked, unexpected_heavy_imports=unexpected,
              feature_order_matches_existing_core=True, feature_count=len(expected),
              features=expected, source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              candidate_source_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
              source_commit=ns["SOURCE_COMMIT"], weight_revision=ns["WEIGHT_REVISION"],
              data_reads=0, weight_reads=0, installs=0, model_fits=0, model_predictions=0,
              family_registration=None)
print(json.dumps(result, ensure_ascii=False, indent=2))

