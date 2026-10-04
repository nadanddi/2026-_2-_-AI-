"""family20 TabDPT 전체 독립 검산. 기본 실행은 비활성, --synthetic은 합성만.
--verify는 부모가 완료 뒤 실행한다. fit/predict/LOG/KKT/원시 train_y/EL1/test 없음.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys

sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
ROOT = H.parents[3]
OUT = ROOT / "집/코덱스/local/ec_tabdpt_20261004_v1"
BASE = ROOT / "집/코덱스/local/ec_tabpfn35_20261003_v1"
OLD = ROOT / "집/코덱스/local/ec_dc4_integration_20261002_v1"
PREP = H.parent / "ec_tabdpt_preparation_20261004_v1"
SITE = ROOT / "집/코덱스/local/tabdpt130_cpu_v1/site"
SEEDS = (7, 101, 2024)
VALIDATORS = ("DIAG10", "A", "B", "EXT10", "EXT12")
FOLD_KEYS = tuple([(v, k) for v, n in
                   (("DIAG10", 10), ("A", 5), ("B", 5), ("EXT10", 1), ("EXT12", 1))
                  for k in range(n)])
FAMILY = 20
ALPHA = .025 / FAMILY
RAW_ATOL = 1e-6
FINAL_ATOL = 2e-7
SOURCE_COMMIT = "97e5494431e9527c7edb31cb4dcfc5f00b232fdf"
WEIGHT_SHA = "97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb"
EXPECTED_DEPS = {
    "core": "057ff4d8da6f3af29105b049251498f79f7fcd6b88d980a8222e5629eb5ce3b2",
    "season": "7d58feeb6a0e7653796a6775f762b659cab3e64078b299d36b52ae69faeff162",
    "env": "82ca7b4bda2e9f50069b53b92d8418a7c0d4dd5ce7127fade16dd6ea78a8dcd2",
    "support": "82074e7a04ea681bc94e94e1c90354a743609e0f008b1a378a5a9a23575debed",
    "adapter": "db919f630d0993e4a8d9bf547524d02563d481536a93ca61628f733f1dd7c97f",
    "runtime_probe": "b4392835d3681f1b24ccef90fbefb6d45ddea73019e82ffa3f6d09ba5e433506",
    "run": "136f72ce14cf7afc078185f0e1c421052910153d6f156ebe8eed604cc3a6d2f2",
}
CORE_RUNTIME = dict(python="3.12.14", numpy="2.5.3", pandas="3.0.1",
                    sklearn="1.9.1", lightgbm="4.7.0")
PIN5 = {"tabdpt": "1.3.0", "faiss-cpu": "1.12.0", "omegaconf": "2.3.0",
        "antlr4-python3-runtime": "4.9.3", "huggingface-hub": "0.36.0"}
CSV_COLUMNS = ["row_id", "farm", "day", "hour", "y", "baseline", "candidate",
               "new_tabdpt_raw", "old_pfn_raw", "r3_raw", "validator", "fold",
               "seed", "clip_lo", "clip_hi"]
np = pd = env = S = None
CHECKS = 0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readj(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def array_sha(x):
    x = np.asarray(x)
    return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


def ids_sha(x):
    return hashlib.sha256("\n".join(map(str, x)).encode()).hexdigest()


def bootstrap_runtime(with_support):
    global np, pd, env, S
    sys.path.insert(0, str(ROOT / "집/클로드/research"))
    import env as e  # 첫 project import
    import numpy as n
    import pandas as p
    np, pd, env = n, p, e
    if with_support:
        sys.path.insert(0, str(H.parent / "statistical_experiments_20261003_v1"))
        import support as support_module
        S = support_module
        assert sha(Path(S.__file__)) == EXPECTED_DEPS["support"]


def compare(a, b, tol=1e-12):
    global CHECKS
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and a.size > 0
    assert np.isfinite(a).all() and np.isfinite(b).all()
    gap = float(np.max(np.abs(a - b)))
    assert gap <= tol, (gap, tol)
    CHECKS += a.size
    return gap


def rmse_both(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    assert y.shape == prediction.shape and y.ndim == 1 and len(y)
    assert np.isfinite(y).all() and np.isfinite(prediction).all()
    scalar = math.sqrt(math.fsum((float(a) - float(b)) ** 2
                                 for a, b in zip(y, prediction)) / len(y))
    vector = float(np.sqrt(np.mean((y - prediction) ** 2)))
    compare([scalar], [vector])
    return scalar, vector


def prefix_scalar(frame, r3, tabdpt, lo, hi):
    """core.shrink를 호출하지 않는 순서 독립 scalar 현재·과거 계산."""
    assert len(frame) == len(r3) == len(tabdpt) and math.isfinite(lo) and math.isfinite(hi) and lo <= hi
    groups = {}
    for i, (farm, day, hour) in enumerate(frame[["farm", "day", "hour"]].itertuples(index=False, name=None)):
        groups.setdefault((str(farm), int(day)), []).append((int(hour), i))
    output = [None] * len(frame)
    for entries in groups.values():
        assert len({h for h, _ in entries}) == len(entries)
        history = []
        for _, i in sorted(entries):
            raw = math.fsum((.8 * float(r3[i]), .2 * float(tabdpt[i])))
            assert math.isfinite(raw)
            history.append(raw)
            value = math.fsum((.5 * raw, .5 * math.fsum(history) / len(history)))
            output[i] = min(float(hi), max(float(lo), value))
    return np.asarray(output, float)


def boot_diag(d, seed):
    """family19와 동일: farm별 정렬 5일 비중첩 block, 20,000회, 고정 RNG."""
    assert set(d.farm) == {"F13", "F47"}
    assert not d.duplicated(["farm", "day", "hour"]).any()
    daily = {}
    for r in d.itertuples():
        delta = (float(r.y) - float(r.candidate)) ** 2 - (float(r.y) - float(r.baseline)) ** 2
        daily.setdefault((r.farm, int(r.day)), []).append(delta)
    rng = np.random.default_rng(20261003 + seed)
    ss, nn = np.zeros(20000), np.zeros(20000)
    block_counts = {}
    for farm in ("F13", "F47"):
        days = sorted(day for f, day in daily if f == farm)
        assert days
        sums, counts = [], []
        for i in range(0, len(days), 5):
            vals = [x for day in days[i:i + 5] for x in daily[farm, day]]
            sums.append(math.fsum(vals)); counts.append(len(vals))
        sums, counts = np.array(sums), np.array(counts)
        block_counts[farm] = len(sums)
        ix = rng.integers(len(sums), size=(20000, len(sums)))
        ss += sums[ix].sum(1); nn += counts[ix].sum(1)
    sample = ss / nn

    # 독립 pandas 일별 합계/위치 block으로 같은 draw 재계산.
    gd = d.assign(delta=(d.y-d.candidate)**2-(d.y-d.baseline)**2).groupby(["farm", "day"]).delta.agg(["sum", "count"])
    rng = np.random.default_rng(20261003 + seed)
    ss2, nn2 = np.zeros(20000), np.zeros(20000)
    for farm in ("F13", "F47"):
        g = gd.loc[farm].sort_index()
        sums = np.array([g["sum"].iloc[i:i+5].sum() for i in range(0, len(g), 5)])
        counts = np.array([g["count"].iloc[i:i+5].sum() for i in range(0, len(g), 5)])
        ix = rng.integers(len(sums), size=(20000, len(sums)))
        ss2 += sums[ix].sum(1); nn2 += counts[ix].sum(1)
    vector_sample = ss2 / nn2
    gap = compare(sample, vector_sample)
    # 0 주변 부동소수점 차이가 채택 판정을 뒤집으면 중단.
    assert np.array_equal(sample >= 0, vector_sample >= 0)
    adjusted = np.quantile(sample, [ALPHA, 1-ALPHA]).tolist()
    ordinary = np.quantile(sample, [.025, .975]).tolist()
    return dict(p_worse=float((sample >= 0).mean()), ci=adjusted,
                ci_family20=adjusted, ci95=ordinary, alpha=ALPHA,
                adjusted_quantiles=[ALPHA, 1-ALPHA], n_bootstrap=20000,
                rng_seed=20261003+seed, blocks_per_farm=block_counts,
                independent_sample_maxdiff=gap)


def validate_first(first, signature, tr, q):
    assert first["status"] == "PASS" and first["signature"] == signature
    assert first["raw_atol"] == RAW_ATOL and first["final_atol"] == FINAL_ATOL
    assert set(first["raw_errors"]) == {"repeat", "single", "reversed", "other_query", "fresh_fit"}
    assert all(math.isfinite(x) and 0 <= x <= RAW_ATOL for x in first["raw_errors"].values())
    assert math.isfinite(first["scalar_maxdiff"]) and 0 <= first["scalar_maxdiff"] <= 1e-12
    expected = {(f, h) for f in ("F13", "F47") for h in (0, 6, 12)}
    for name, reference in (("prefix", q), ("feature_causal", tr)):
        records = first[name]
        assert len(records) == 6 and {(r["farm"], r["hour"]) for r in records} == expected
        for r in records:
            assert r["day"] == int(reference.loc[reference.farm == r["farm"], "day"].min())
            fields = {"raw_maxdiff": RAW_ATOL, "final_maxdiff": FINAL_ATOL} if name == "prefix" else {"feature_maxdiff": 1e-12}
            for field, tol in fields.items():
                assert math.isfinite(r[field]) and 0 <= r[field] <= tol


def check_runtime(provenance, dependencies, input_sha):
    expected_keys = {"run_sha256", "support_sha256", "adapter_sha256",
                     "runtime_probe_sha256", "weight_sha256", "input_sha256", "runtime"}
    assert set(provenance) == expected_keys
    assert provenance["run_sha256"] == dependencies["run"]
    assert provenance["support_sha256"] == dependencies["support"]
    assert provenance["adapter_sha256"] == dependencies["adapter"]
    assert provenance["runtime_probe_sha256"] == dependencies["runtime_probe"]
    assert provenance["weight_sha256"] == WEIGHT_SHA and provenance["input_sha256"] == input_sha
    runtime = provenance["runtime"]
    reference = readj(PREP / "runtime_probe_result_v3.json")
    assert runtime == reference
    assert runtime["status"] == "PASS_IMPORT_ONLY_FIT_PREDICT_UNTESTED"
    assert runtime["source_commit"] == SOURCE_COMMIT and runtime["exact_five_pins"] == PIN5
    assert all(runtime["versions"][name] == value for name, value in PIN5.items())
    assert runtime["module_paths"]["torch"]["module_version"] == "2.14.0+cpu"
    assert runtime["module_paths"]["safetensors"]["module_version"] == "0.8.0"
    assert len(runtime["source_sha256"]) == 7
    for relative, expected in runtime["source_sha256"].items():
        assert sha(SITE / relative) == expected
    assert all(runtime[name] == 0 for name in ("new_installs", "weight_reads", "new_weight_downloads", "data_reads", "fit", "predict"))
    return runtime


def calculate_scores(frames):
    scores = []
    for v in VALIDATORS:
        for seed in SEEDS:
            g = frames[(frames.validator == v) & (frames.seed == seed)]
            rb, nb = rmse_both(g.y, g.baseline)
            rc, nc = rmse_both(g.y, g.candidate)
            assert (rc < rb) == (nc < nb), "두 RMSE 계산의 엄격 방향이 다릅니다"
            scores.append(dict(validator=v, seed=seed, n=len(g), baseline=rb,
                               candidate=rc, baseline_numpy=nb, candidate_numpy=nc,
                               delta_rmse=rc-rb, change_pct=100*(rc/rb-1)))
    scores = pd.DataFrame(scores)
    assert len(scores) == 15 and not scores.duplicated(["validator", "seed"]).any()
    boots, segments = {}, []
    for seed in SEEDS:
        d = frames[(frames.validator == "DIAG10") & (frames.seed == seed)].copy()
        assert len(d) == 8640 and len(d[["farm", "day"]].drop_duplicates()) == 360
        hours = d.groupby(["farm", "day"]).hour.agg(lambda x: set(x))
        assert all(x == set(range(24)) for x in hours)
        high = d.groupby(["farm", "day"]).y.transform("mean") >= 1
        assert len(d.loc[high, ["farm", "day"]].drop_duplicates()) == 31
        assert len(d.loc[~high, ["farm", "day"]].drop_duplicates()) == 329
        assert int(high.sum()) == 31 * 24
        boots[seed] = boot_diag(d, seed)
        for label, mask in (("high", high), ("ordinary", ~high),
                            ("late", d.day >= 179), ("F13", d.farm == "F13"),
                            ("F47", d.farm == "F47"), ("hour0", d.hour == 0)):
            g = d[mask]
            rb, _ = rmse_both(g.y, g.baseline); rc, _ = rmse_both(g.y, g.candidate)
            bb = math.fsum(float(x) for x in g.baseline-g.y) / len(g)
            cb = math.fsum(float(x) for x in g.candidate-g.y) / len(g)
            compare([bb, cb], [(g.baseline-g.y).mean(), (g.candidate-g.y).mean()])
            segments.append(dict(seed=seed, segment=label, n=len(g),
                                 days=len(g[["farm", "day"]].drop_duplicates()),
                                 baseline=rb, candidate=rc,
                                 baseline_bias=bb, candidate_bias=cb))
    passed = bool((scores.delta_rmse < 0).all() and
                  all(x["p_worse"] < ALPHA and x["ci_family20"][1] < 0 for x in boots.values()))
    return scores, pd.DataFrame(segments), boots, passed


def verify_complete():
    destinations = [H/"full_verification_v1.json", H/"full_scores_v1.csv", H/"full_segments_v1.csv"]
    assert all(not p.exists() for p in destinations), "기존 결과는 보존하고 새 버전 verifier를 사용하세요"
    assert (H/"preregistration_v6.md").is_file(), "조정 CI 정정 사전등록 v6 확인 필요"
    prepared = readj(H/"preparation_v4.json")
    fit = readj(H/"fit_audit_v1.json")
    assert prepared["status"] == "PASS" and prepared["fit_count"] == 0
    assert prepared["runtime"] == CORE_RUNTIME and prepared["dependencies"] == EXPECTED_DEPS
    assert fit["status"] == "PASS" and fit["family"] == FAMILY and fit["preparation"] == prepared
    assert len(fit["cells"]) == 66
    assert tuple((x["validator"], x["fold"]) for x in prepared["manifest"]) == FOLD_KEYS
    # 최초 감사 및 source 관문을 통과하기 전 score를 계산하지 않는다.
    first_path = H/"first_fold_verification_v1.json"
    first = readj(first_path)
    bootstrap_runtime(with_support=True)
    lab, core, wv, folds, outer = S.loadec()  # 공개 OOF target만; 원시 EC y/test/EL1 호출 없음
    assert tuple((v, k) for v, k, _, _ in folds) == FOLD_KEYS
    assert len(lab) == 8640 and lab.row_id.is_unique
    assert set(lab.farm) == {"F13", "F47"}
    columns = [c for c in core.FULL if c != "day"] + ["season"]
    assert len(columns) == 38 and columns[-1] == "season" and "day" not in columns
    dependencies = dict(core=sha(Path(core.__file__)),
                        season=sha(Path(sys.modules[S.mapping.__module__].__file__)),
                        env=sha(Path(env.__file__)), support=sha(Path(S.__file__)),
                        adapter=sha(PREP/"adapter_draft_v1.py"),
                        runtime_probe=sha(PREP/"runtime_probe_v3.py"), run=sha(H/"run_v4.py"))
    assert dependencies == EXPECTED_DEPS
    import platform, sklearn, lightgbm
    runtime_core = dict(python=platform.python_version(), numpy=np.__version__,
                        pandas=pd.__version__, sklearn=sklearn.__version__, lightgbm=lightgbm.__version__)
    assert runtime_core == CORE_RUNTIME
    input_sha = sha(Path(env.DATA)/"train_X.csv")
    public_sha = sha(OLD/"v2_integration_oof.csv")
    assert prepared["input_sha256"] == input_sha and prepared["public_cache_sha256"] == public_sha
    public = outer[(outer.validator == "DIAG10") & (outer.seed == 7)]
    assert len(public) == 8640 and public.row_id.is_unique
    label_map = public.set_index("row_id").sub_ec
    compare(lab.sub_ec, label_map.reindex(lab.row_id))
    provenance = first["signature"]["provenance"]
    runtime = check_runtime(provenance, dependencies, input_sha)
    receipt = readj(H/"weight_receipt_v1.json")
    assert receipt["status"] == "PASS" and receipt["sha256"] == WEIGHT_SHA
    assert receipt["bytes"] == 252233296 and receipt["revision"] == "a5ca6e01c0fa09ec68c73e958e5199d1932abb3a"
    install = readj(H/"install_report_v1.json")
    installed = {x["metadata"]["name"]: x for x in install["install"]}
    assert set(installed) == set(PIN5)
    assert all(installed[n]["metadata"]["version"] == p for n, p in PIN5.items())
    assert installed["tabdpt"]["download_info"]["vcs_info"]["commit_id"] == SOURCE_COMMIT

    expected_paths = {OUT/f"{v}_{k}_{s}_pred.csv" for v, k in FOLD_KEYS for s in SEEDS}
    assert set(OUT.glob("*_pred.csv")) == expected_paths
    expected_meta = {OUT/f"{v}_{k}_{s}.json" for v, k in FOLD_KEYS for s in SEEDS}
    assert set(OUT.glob("*.json")) == expected_meta
    frames, cells, split_audit = [], [], []
    scalar_maxdiff = 0.
    rebuilt_manifest = []
    for i, (v, k, tm, vm) in enumerate(folds):
        tr, q = S.seasonal(lab[tm], lab[vm], wv)
        tr, q = tr.reset_index(drop=True), q.reset_index(drop=True)
        assert len(tr) > 512 and not set(tr.row_id) & set(q.row_id)
        assert not np.isnan(tr[columns].to_numpy(float)).all(axis=0).any()
        keys_tr = set(zip(tr.farm, tr.day)); keys_q = set(zip(q.farm, q.day))
        assert not keys_tr & keys_q
        assert all(not (f == ff and abs(day-qd) <= 1)
                   for f, day in keys_tr for ff, qd in keys_q)
        cache_path = BASE/f"{v}_{k}_baseline.npz"
        old = dict(np.load(cache_path, allow_pickle=False))
        assert np.array_equal(old["row_id"], q.row_id)
        compare([old["lo"], old["hi"]], [tr.sub_ec.min(), tr.sub_ec.max()])
        pfn = []
        for context in (1, 2, 3, 4):
            bag = dict(np.load(OLD/f"{v}_{k}_pfn_{context}.npz", allow_pickle=False))
            assert np.array_equal(bag["row_id"], q.row_id)
            assert np.array_equal(bag["context_row_id"], old[f"context_{context}"])
            assert set(bag["context_row_id"]) <= set(tr.row_id)
            compare(bag["sub_ec"], q.sub_ec)
            pfn.append(bag["raw_pfn"])
        compare(old["old_pfn_raw"], np.mean(pfn, axis=0))
        sig = dict(validator=v, fold=k, runtime=runtime_core, features=columns,
                   dependencies=dependencies, input_sha256=input_sha, public_cache_sha256=public_sha,
                   train_ids=ids_sha(tr.row_id), query_ids=ids_sha(q.row_id),
                   train_features=array_sha(tr[columns].to_numpy(float)),
                   query_features=array_sha(q[columns].to_numpy(float)),
                   train_targets=array_sha(tr.sub_ec.to_numpy(float)),
                   query_targets=array_sha(q.sub_ec.to_numpy(float)),
                   bounds=[float(old["lo"]), float(old["hi"])], baseline_sha256=sha(cache_path))
        assert sig == prepared["manifest"][i]
        rebuilt_manifest.append(sig)
        split_audit.append(dict(validator=v, fold=k, train_days=len(keys_tr),
                                query_days=len(keys_q), query_order_sha256=ids_sha(q.row_id)))
        for seed in SEEDS:
            path = OUT/f"{v}_{k}_{seed}_pred.csv"
            meta = readj(OUT/f"{v}_{k}_{seed}.json")
            signature = dict(**sig, seed=seed, provenance=provenance)
            assert meta["status"] == "PASS" and meta["signature"] == signature
            assert meta["csv_sha256"] == sha(path)
            assert math.isfinite(meta["fit_predict_seconds"]) and meta["fit_predict_seconds"] >= 0
            if (v, k, seed) == ("DIAG10", 0, 7):
                assert meta["first_audit_sha256"] == sha(first_path)
                validate_first(first, signature, tr, q)
            else:
                assert "first_audit_sha256" not in meta
            original_meta = readj(OLD/f"{v}_{k}_r3_{seed}.json")
            assert original_meta["provenance"]["shared"]["input_sha256"]["train_X.csv"] == input_sha
            assert original_meta["provenance"]["shared"]["core_sha256"] == dependencies["core"]
            assert all(original_meta["provenance"]["environment"][n] == value for n, value in runtime_core.items())
            original_r3 = dict(np.load(OLD/f"{v}_{k}_r3_{seed}.npz", allow_pickle=False))
            assert np.array_equal(original_r3["row_id"], q.row_id)
            compare(old[f"r3_{seed}"], original_r3["raw_r3"])
            d = pd.read_csv(path, float_precision="round_trip")
            assert list(d.columns) == CSV_COLUMNS
            assert np.array_equal(d.row_id, q.row_id) and d.row_id.is_unique
            assert (d.validator == v).all() and (d.fold == k).all() and (d.seed == seed).all()
            for c in ("farm", "day", "hour"):
                assert np.array_equal(d[c], q[c])
            compare(d.y, q.sub_ec)
            compare(d.y, label_map.reindex(d.row_id))
            baseline = outer[(outer.validator == v) & (outer.validation_fold == k) &
                             (outer.seed == seed)].set_index("row_id").season_v2.reindex(q.row_id).to_numpy()
            compare(d.baseline, baseline); compare(d.baseline, old[f"baseline_{seed}"])
            compare(d.r3_raw, old[f"r3_{seed}"]); compare(d.old_pfn_raw, old["old_pfn_raw"])
            compare(d.clip_lo, np.repeat(old["lo"], len(q))); compare(d.clip_hi, np.repeat(old["hi"], len(q)))
            compare(d.baseline, np.clip(core.shrink(.8*d.r3_raw.to_numpy()+.2*d.old_pfn_raw.to_numpy(), q), old["lo"], old["hi"]))
            compare(d.candidate, np.clip(core.shrink(.8*d.r3_raw.to_numpy()+.2*d.new_tabdpt_raw.to_numpy(), q), old["lo"], old["hi"]))
            scalar = prefix_scalar(q, d.r3_raw.to_numpy(), d.new_tabdpt_raw.to_numpy(), float(old["lo"]), float(old["hi"]))
            scalar_maxdiff = max(scalar_maxdiff, compare(scalar, d.candidate))
            scalar_base = prefix_scalar(q, d.r3_raw.to_numpy(), d.old_pfn_raw.to_numpy(), float(old["lo"]), float(old["hi"]))
            compare(scalar_base, d.baseline)
            cells.append(meta); frames.append(d)
    assert cells == fit["cells"] and rebuilt_manifest == prepared["manifest"]
    concatenated = pd.concat(frames, ignore_index=True)
    aggregate = pd.read_csv(OUT/"oof.csv", float_precision="round_trip")
    assert len(concatenated) == len(aggregate) == 83160
    assert list(aggregate.columns) == CSV_COLUMNS
    assert not aggregate.duplicated(["validator", "fold", "seed", "row_id"]).any()
    for c in CSV_COLUMNS:
        if c in ("row_id", "farm", "validator", "day", "hour", "fold", "seed"):
            assert np.array_equal(aggregate[c], concatenated[c])
        else:
            compare(aggregate[c], concatenated[c])
    assert set(zip(aggregate.validator, aggregate.fold, aggregate.seed)) == {(v, k, s) for v, k in FOLD_KEYS for s in SEEDS}
    scores, segments, boots, passed = calculate_scores(concatenated)
    # 모든 감사 완료 후에만 새 출력 생성. 기존 파일은 덮어쓰지 않는다.
    for table, target in ((scores, destinations[1]), (segments, destinations[2])):
        with target.open("x", encoding="utf-8", newline="") as handle:
            table.to_csv(handle, index=False)
    result = dict(status="PASS", decision="PUBLIC_PASS_PENDING_FURTHER_GUARD" if passed else "REJECT",
                  public_pass=passed, family=FAMILY, alpha=ALPHA,
                  adoption_ci_quantiles=[ALPHA, 1-ALPHA], rows=len(concatenated), cells=66,
                  score_cells=15, checks=CHECKS, bootstrap=boots, scalar_maxdiff=scalar_maxdiff,
                  split_audit=split_audit, source_sha256=dependencies["run"],
                  verifier_sha256=sha(Path(__file__)), dependencies=dependencies,
                  preparation_sha256=sha(H/"preparation_v4.json"), fit_audit_sha256=sha(H/"fit_audit_v1.json"),
                  first_audit_sha256=sha(first_path), aggregate_sha256=sha(OUT/"oof.csv"),
                  preregistration_v6_sha256=sha(H/"preregistration_v6.md"),
                  scores_sha256=sha(destinations[1]), segments_sha256=sha(destinations[2]),
                  runtime=runtime, public_cache_sha256=public_sha, input_sha256=input_sha,
                  verification_fit=0, verification_predict=0,
                  limitations=["repeated public validation, not untouched holdout",
                               "no raw train_y, locked target, EL1/test scoring",
                               "source/manifest replay does not prove every fold season feature causal independently",
                               "saved audit verification does not rerun TabDPT predictions",
                               "runtime metadata/source hashes and weight receipt checked; checkpoint bytes not reread",
                               "preregistration timing relies on parent git registration; runner signatures omit prereg hash"])
    with destinations[0].open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(scores.to_string(index=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ("runtime", "split_audit")},
                     ensure_ascii=False, indent=2))


def synthetic_check():
    bootstrap_runtime(with_support=False)
    # 실제 core의 shrink 함수만 AST로 추출. module import나 data loader 호출 없음.
    source = ROOT/"집/코덱스/analysis/codex_independent/rl_ec_v1/run.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "shrink")
    namespace = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    frame = pd.DataFrame(dict(farm=["F47", "F13", "F13", "F47", "F13"],
                              day=[2, 1, 1, 2, 1], hour=[1, 2, 0, 0, 1]))
    r3, dpt = np.array([5., 2., 1., 0., 0.]), np.array([-5., 6., 0., 0., 10.])
    vector = np.clip(namespace["shrink"](.8*r3+.2*dpt, frame), .1, 2.)
    scalar = prefix_scalar(frame, r3, dpt, .1, 2.)
    compare(scalar, [2., 2., .8, .1, 1.7])
    compare(scalar, vector)
    for i, row in frame.iterrows():
        mask = (frame.farm == row.farm) & (frame.day == row.day) & (frame.hour <= row.hour)
        prior = prefix_scalar(frame[mask], r3[mask], dpt[mask], .1, 2.)
        compare(prior[-1:] if len(prior) == 1 else
                [prior[list(frame.index[mask]).index(i)]], [scalar[i]])
    rmse_both([0., 1., 3.], [.1, .7, 2.8])
    rows = []
    for farm in ("F13", "F47"):
        for day in range(12):
            for hour in range(24):
                y = .5 + day/100 + hour/1000
                rows.append(dict(farm=farm, day=day, hour=hour, y=y,
                                 baseline=y+.2, candidate=y+.1))
    boot = boot_diag(pd.DataFrame(rows), 7)
    assert boot["p_worse"] == 0 and boot["ci_family20"][1] < 0 and boot["ci95"][1] < 0
    equal = pd.DataFrame(rows); equal["candidate"] = equal["baseline"]
    tied = boot_diag(equal, 7)
    assert tied["p_worse"] == 1 and tied["ci_family20"] == [0., 0.]
    result = dict(status="PASS_SYNTHETIC_ONLY", checks=CHECKS, source_shrink_hash=sha(source),
                  family=FAMILY, alpha=ALPHA, adjusted_quantiles=[ALPHA, 1-ALPHA],
                  toy_blocks_per_farm=boot["blocks_per_farm"], model_fit=0, model_predict=0,
                  actual_data_reads=0, actual_score_reads=0, weight_reads=0)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--verify", action="store_true")
    arg = parser.parse_args()
    if arg.synthetic:
        synthetic_check()
    elif arg.verify:
        verify_complete()
    else:
        print(json.dumps(dict(status="DRAFT_NO_RESULT_OR_DATA_READS", family=FAMILY, alpha=ALPHA,
                              expected_cells=66, expected_score_cells=15,
                              actual_data_reads=0, actual_score_reads=0), indent=2))

