"""Family20 v4: AST-only delta audit and actual-guard synthetic checks.

No experiment imports, real data, weights, model fit or prediction.
Created synthetic artifacts are preserved; this check is deliberately one-shot.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('C:/work/farmai')
HERE = ROOT / '집/코덱스/analysis/ec_harsh_review_20261004_v1'
SOURCE = ROOT / '집/코덱스/analysis/ec_tabdpt_20261004_v1'
RESULT = HERE / 'tabdpt_v4_aggregate_check_v1.json'
SYNTHETIC = HERE / 'tabdpt_v4_synthetic_aggregate_v1'
assert __debug__, 'Run guard audit without -O.'
assert not RESULT.exists() and not SYNTHETIC.exists(), 'Preserve previous audit.'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main_node(tree):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')


def dump(node):
    return ast.dump(node, include_attributes=False)


old_tree = ast.parse((SOURCE / 'run_v3.py').read_text(encoding='utf-8-sig'))
new_tree = ast.parse((SOURCE / 'run_v4.py').read_text(encoding='utf-8-sig'))
old_main = main_node(old_tree)
new_main = main_node(new_tree)
normal = copy.deepcopy(new_tree)
normal_main = main_node(normal)
messages = ['Aggregate already exists: preserve and verify separately.', 'Preserve existing aggregate.']
aggregate_asserts = [n for n in new_main.body if isinstance(n, ast.Assert)
                     and isinstance(n.msg, ast.Constant) and n.msg.value in messages]
assert len(aggregate_asserts) == 2
assert [n.msg.value for n in aggregate_asserts] == messages
for guard in aggregate_asserts:
    assert ast.unparse(guard.test) == "not (OUT / 'oof.csv').exists()"
for n in list(normal_main.body):
    if isinstance(n, ast.Assert) and isinstance(n.msg, ast.Constant) and n.msg.value in messages:
        normal_main.body.remove(n)
save_with = [n for n in new_main.body if isinstance(n, ast.With)]
assert len(save_with) == 1
save_with = save_with[0]
assert ast.unparse(save_with.items[0].context_expr) == "(OUT / 'oof.csv').open('x', encoding='utf-8', newline='')"
assert ast.unparse(save_with.items[0].optional_vars) == 'handle'
assert len(save_with.body) == 1
assert ast.unparse(save_with.body[0]) == 'pd.concat(outputs, ignore_index=True).to_csv(handle, index=False)'
old_save = next(n for n in old_main.body if isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                and n.value.func.attr == 'to_csv')
normal_with = next(n for n in normal_main.body if isinstance(n, ast.With))
normal_main.body[normal_main.body.index(normal_with)] = copy.deepcopy(old_save)
name_count = 0
for n in ast.walk(normal):
    if isinstance(n, ast.Constant) and n.value == 'preparation_v4.json':
        n.value = 'preparation_v3.json'
        name_count += 1
assert name_count == 2
assert dump(normal) == dump(old_tree)

prepared = json.loads((SOURCE / 'preparation_v4.json').read_text(encoding='utf-8'))
source_sha = sha(SOURCE / 'run_v4.py')
assert prepared['status'] == 'PASS' and prepared['folds'] == 22 and prepared['fit_count'] == 0
assert len(prepared['manifest']) == 22
assert prepared['dependencies']['run'] == source_sha
assert all(m['dependencies']['run'] == source_sha for m in prepared['manifest'])
doc6 = (SOURCE / 'preregistration_v6.md').read_text(encoding='utf-8-sig')
assert '[0.00125, 0.99875]' in doc6 and '사전v5의 95% CI 표기는 작성 오류' in doc6

start_guard, final_guard = aggregate_asserts
index = new_main.body.index(start_guard)
assert index == 5
runtime_import = new_main.body[index + 1]
assert ast.unparse(runtime_import) == "R = module('tabdpt_import_guard', PREP / 'runtime_probe_v3.py')"
runtime_probe = new_main.body[index + 2]
assert ast.unparse(runtime_probe) == 'imports = R.import_probe(SITE)'
assert start_guard.lineno < runtime_import.lineno < runtime_probe.lineno
assert final_guard.lineno < save_with.lineno

# Execute the real main prefix with a stub loader/preflight/parser.
# Prefix stops before runtime module import; no experiment module is imported.
prefix_fn = copy.deepcopy(new_main)
prefix_fn.name = 'actual_prefix_with_stubs'
prefix_fn.body = prefix_fn.body[:index + 1]
prefix_fn.body.append(ast.Return(value=ast.Constant('GUARDS_PASSED')))
prefix_code = compile(ast.fix_missing_locations(ast.Module(body=[prefix_fn], type_ignores=[])),
                      '<actual-v4-prefix-stubbed>', 'exec')


class FakeParser:
    def add_argument(self, *args, **kwargs):
        pass

    def parse_args(self):
        return SimpleNamespace(prepare=False)


class StubH:
    def __truediv__(self, name):
        if name == 'preparation_v4.json':
            return SimpleNamespace(read_text=lambda **kw: json.dumps({'stub': True}))
        if name == 'fit_audit_v1.json':
            return SimpleNamespace(exists=lambda: False)
        raise AssertionError('Unexpected real path request: ' + name)


SYNTHETIC.mkdir()
cases = []


def folder(name):
    p = SYNTHETIC / name
    p.mkdir()
    return p


def namespace(out):
    calls = []
    ns = {'argparse': SimpleNamespace(ArgumentParser=FakeParser), 'json': json,
          'H': StubH(), 'OUT': out,
          'S': SimpleNamespace(loadec=lambda: (None, None, None, None, None)),
          'preflight': lambda *args: (None, None, {'stub': True})}
    exec(prefix_code, ns)
    return ns, calls


absent = folder('start_absent')
ns, _ = namespace(absent)
assert ns['actual_prefix_with_stubs']() == 'GUARDS_PASSED'
assert not (absent / 'oof.csv').exists()
cases.append({'case': 'start_absent', 'status': 'PASS', 'imports_fit_predict': 0})

present = folder('start_present')
sentinel = b'preserved,synthetic\r\n1,2\r\n'
(present / 'oof.csv').write_bytes(sentinel)
before = sha(present / 'oof.csv')
ns, _ = namespace(present)
try:
    ns['actual_prefix_with_stubs']()
except AssertionError as e:
    assert str(e) == messages[0]
else:
    raise AssertionError('Existing aggregate did not stop prefix.')
assert sha(present / 'oof.csv') == before
cases.append({'case': 'start_existing_stops_before_runtime_import', 'status': 'PASS', 'preserved_sha256': before})

guard_code = compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(final_guard)], type_ignores=[])),
                     '<actual-v4-final-assert>', 'exec')
save_code = compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(save_with)], type_ignores=[])),
                    '<actual-v4-exclusive-save>', 'exec')


class SyntheticDataFrame:
    def to_csv(self, handle, index):
        assert index is False
        handle.write('synthetic,value\nrow,1\n')


def concat_stub(outputs, ignore_index):
    assert outputs == ['SYNTHETIC_ONLY'] and ignore_index is True
    return SyntheticDataFrame()


def save_namespace(out):
    return {'OUT': out, 'outputs': ['SYNTHETIC_ONLY'], 'pd': SimpleNamespace(concat=concat_stub)}


final_absent = folder('final_absent')
ns = save_namespace(final_absent)
exec(guard_code, ns)
exec(save_code, ns)
assert (final_absent / 'oof.csv').read_bytes() == b'synthetic,value\nrow,1\n'
cases.append({'case': 'final_absent_exclusive_creates', 'status': 'PASS', 'new_sha256': sha(final_absent / 'oof.csv')})

final_present = folder('final_present')
(final_present / 'oof.csv').write_bytes(sentinel)
before = sha(final_present / 'oof.csv')
ns = save_namespace(final_present)
try:
    exec(guard_code, ns)
except AssertionError as e:
    assert str(e) == messages[1]
else:
    raise AssertionError('Final assert accepted existing aggregate.')
assert sha(final_present / 'oof.csv') == before
cases.append({'case': 'final_existing_assert_rejects', 'status': 'PASS', 'preserved_sha256': before})

try:
    exec(save_code, ns)
except FileExistsError:
    pass
else:
    raise AssertionError('Exclusive open overwrote existing aggregate.')
assert sha(final_present / 'oof.csv') == before
cases.append({'case': 'exclusive_alone_rejects_existing', 'status': 'PASS', 'preserved_sha256': before})

race = folder('appeared_after_assert')
ns = save_namespace(race)
exec(guard_code, ns)
(race / 'oof.csv').write_bytes(sentinel)
before = sha(race / 'oof.csv')
try:
    exec(save_code, ns)
except FileExistsError:
    pass
else:
    raise AssertionError('Creation race overwrote aggregate.')
assert sha(race / 'oof.csv') == before
cases.append({'case': 'file_appears_between_assert_and_open', 'status': 'PASS', 'preserved_sha256': before})

result = {'status': 'PASS', 'source_v3_sha256': sha(SOURCE / 'run_v3.py'),
          'source_v4_sha256': source_sha, 'normalized_full_ast_equal': True,
          'allowed_delta': {'new_oof_asserts': messages, 'exclusive_open_mode': 'x',
                            'prepare_name_changed_constants': name_count},
          'prepare_source_hash_matches_all_22': True, 'prepare_fit_count': prepared['fit_count'],
          'preregistration_v6_adjusted_ci_correction_confirmed': True,
          'first_guard_precedes_runtime_import': True, 'python_debug_guards_enabled': __debug__,
          'synthetic_cases': cases, 'synthetic_case_count': len(cases),
          'actual_model_fit_count': 0, 'actual_model_predict_count': 0,
          'real_data_or_weight_reads': 0,
          'limits': ['Aggregate creation is exclusive, not an atomic transaction; an interrupted write can leave a partial aggregate that blocks future fitting.',
                     'Existing assertions presume non-optimized Python; audit ran with __debug__ true.',
                     'First audit day and all-fold season causality scopes remain unchanged.']}
with RESULT.open('x', encoding='utf-8') as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False, indent=2))
