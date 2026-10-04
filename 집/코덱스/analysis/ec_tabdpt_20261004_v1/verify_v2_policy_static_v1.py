"""v1/v2의 후보 후처리·통계판정·bootstrap 불변을 AST로 확인. 실제 data/results 0."""
import ast
import hashlib
import json
from pathlib import Path
H = Path(__file__).resolve().parent
old_path, new_path = H/"verify_full_v1.py", H/"verify_full_v2.py"
old = ast.parse(old_path.read_text(encoding="utf-8"))
new = ast.parse(new_path.read_text(encoding="utf-8"))
names = ("prefix_scalar", "rmse_both", "boot_diag", "calculate_scores")
checks = {}
for name in names:
    a = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
    b = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == name)
    checks[name] = ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    assert checks[name]
def constants(tree):
    return {n.targets[0].id: ast.dump(n.value, include_attributes=False)
            for n in tree.body if isinstance(n, ast.Assign) and len(n.targets)==1
            and isinstance(n.targets[0], ast.Name)}
a,b = constants(old),constants(new)
for name in ("SEEDS", "VALIDATORS", "FOLD_KEYS", "FAMILY", "ALPHA",
             "RAW_ATOL", "FINAL_ATOL", "SOURCE_COMMIT", "WEIGHT_SHA", "EXPECTED_DEPS",
             "CORE_RUNTIME", "PIN5", "CSV_COLUMNS"):
    checks[name] = a[name] == b[name]
    assert checks[name]
print(json.dumps(dict(status="PASS_STATIC_POLICY_UNCHANGED", checks=checks,
                      old_verifier_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
                      new_verifier_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),
                      actual_candidate_results_read=0, actual_candidate_scores_read=0,
                      model_fit=0, model_predict=0), ensure_ascii=False, indent=2))

