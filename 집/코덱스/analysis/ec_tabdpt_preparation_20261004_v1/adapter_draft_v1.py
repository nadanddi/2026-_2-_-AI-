"""실행 비활성 초안: TabDPT v1.3.0의 .2 PFN 대체 부품.
기본 실행은 명세 JSON만 출력한다. 데이터 파일을 읽는 경로는 없다.
실제 실행 담당자가 사전등록/환경/가중치 검사를 완료한 뒤 새 버전으로 승격한다.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parents[3]
SOURCE_COMMIT = "97e5494431e9527c7edb31cb4dcfc5f00b232fdf"
WEIGHT_REVISION = "a5ca6e01c0fa09ec68c73e958e5199d1932abb3a"
WEIGHT_FILENAME = "tabdpt1_3.safetensors"
WEIGHT_BYTES = 252233296
WEIGHT_SHA256 = "97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb"
SEEDS = (7, 101, 2024)
RAW_ATOL = 1e-6
FINAL_ATOL = 2e-7

ACTS = ("act_vent", "act_shade", "act_thermal", "act_heating",
        "act_circfan", "act_co2", "act_fog")
INDOOR = ("in_temp", "in_hum", "in_co2")
FULL38 = (INDOOR + ACTS + ("hr_sin", "hr_cos", "midnight")
          + tuple(v + "_h0" for v in ACTS + INDOOR)
          + tuple(v + s for v in ACTS for s in ("_tdm", "_tdz"))
          + ("season",))
CONSTRUCTOR = dict(normalizer="standard", missing_indicators=False,
                   clip_sigma=8.0, feature_reduction="pca",
                   context_reduction="retrieval", faiss_metric="l2",
                   device="cpu", use_flash=False, compile=False, verbose=False)
PREDICT = dict(context_size=512, n_ensembles=8, batch_size=8, output_type="mean")
SPEC = dict(status="DRAFT_NOT_REGISTERED_NOT_EXECUTED", family_count=None,
            replacement=".2 PFN component only", source_commit=SOURCE_COMMIT,
            weight_revision=WEIGHT_REVISION, weight_filename=WEIGHT_FILENAME,
            weight_bytes=WEIGHT_BYTES, weight_sha256=WEIGHT_SHA256,
            seeds=SEEDS, constructor=CONSTRUCTOR, predict=PREDICT,
            features=FULL38, raw_atol=RAW_ATOL, final_atol=FINAL_ATOL,
            model_fit_count=0, training_count=0, download_count=0,
            install_count=0, performance_claim=None)


def require_runtime(execution_authorized: bool) -> None:
    if not execution_authorized:
        raise RuntimeError("준비 초안입니다. 사전등록/환경/가중치 관문 후 새 버전으로 승격하세요.")


def verify_weight(path: Path, *, execution_authorized: bool = False) -> Path:
    require_runtime(execution_authorized)
    path = path.resolve(strict=True)
    if not path.is_file() or path.name != WEIGHT_FILENAME:
        raise ValueError("고정된 로컬 checkpoint 경로가 필요합니다.")
    if path.stat().st_size != WEIGHT_BYTES:
        raise ValueError("고정 weight size 불일치")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != WEIGHT_SHA256:
        raise ValueError("고정 weight SHA256 불일치")
    return path


def activate_runtime(runtime_root: Path, *, execution_authorized: bool = False) -> None:
    """향후 깨끗한 worker에서만 호출. 기존 R3/PFN process를 건드리지 않는다."""
    require_runtime(execution_authorized)
    if any(n in sys.modules for n in ("tabdpt", "faiss", "omegaconf", "huggingface_hub")):
        raise RuntimeError("이미 부품을 import한 process입니다. 별도 worker가 필요합니다.")
    runtime_root = runtime_root.resolve(strict=True)
    allowed_parent = (ROOT / "집" / "코덱스" / "local").resolve()
    if not runtime_root.is_relative_to(allowed_parent):
        raise ValueError("runtime은 집/코덱스/local 아래 분리된 경로여야 합니다.")
    sys.path.insert(0, str(ROOT / "집" / "클로드" / "research"))
    import env  # 프로젝트 실행 시 첫 project import
    import env_extra
    sys.path.insert(0, str(runtime_root))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def verify_package_pin(*, execution_authorized: bool = False) -> dict[str, str]:
    require_runtime(execution_authorized)
    from packaging.version import Version
    from packaging.specifiers import SpecifierSet
    required = {"tabdpt": "==1.3.0", "faiss-cpu": "==1.12.0",
                "huggingface-hub": ">=0.33.2,<2.0", "numpy": ">=1.25,<3",
                "omegaconf": ">=2.1.1,<3", "safetensors": ">=0.5.3,<1",
                "scikit-learn": ">=1.4,<2", "scipy": ">=1.9,<2",
                "torch": ">=2.6,<3", "tqdm": ">=4.38,<5"}
    found = {}
    for name, spec in required.items():
        value = metadata.version(name)
        if Version(value) not in SpecifierSet(spec):
            raise RuntimeError(f"{name}={value}: pinned requirement {spec} 불일치")
        found[name] = value
    direct = metadata.distribution("tabdpt").read_text("direct_url.json")
    commit = json.loads(direct or "{}").get("vcs_info", {}).get("commit_id")
    if commit != SOURCE_COMMIT:
        raise RuntimeError("TabDPT installed source commit을 확인할 수 없습니다.")
    return found


def build_and_fit(
    X_train: Any, y_train: Any, weight_path: Path, seed: int,
    *, execution_authorized: bool = False,
) -> Any:
    """향후 학습 fold 배열만 전달. 파일/전체 정답/평가 목표를 읽지 않는다."""
    require_runtime(execution_authorized)
    verify_package_pin(execution_authorized=True)
    pinned_weight = verify_weight(weight_path, execution_authorized=True)
    import numpy as np
    import torch
    import faiss
    from tabdpt.regressor import TabDPTRegressor

    if seed not in SEEDS:
        raise ValueError("사전 고정 seed만 허용")
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if X_train.ndim != 2 or X_train.shape[1] != len(FULL38):
        raise ValueError("FULL38 순서의 2차원 train 배열 필요")
    if y_train.shape != (len(X_train),) or len(X_train) <= 512:
        raise ValueError("512보다 큰 train 문맥 및 정렬된 1차원 목표 필요")
    if np.isinf(X_train).any() or not np.isfinite(y_train).all():
        raise ValueError("Inf 입력/비유한 목표 불가")
    if np.isnan(X_train).all(axis=0).any():
        raise ValueError("전부 결측인 feature는 SimpleImputer에서 삭제되므로 중단")

    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    for key in CONSTRUCTOR:
        if key not in inspect.signature(TabDPTRegressor.__init__).parameters:
            raise RuntimeError(f"공식 constructor API와 다른 key: {key}")
    for key in PREDICT:
        if key not in inspect.signature(TabDPTRegressor.predict).parameters:
            raise RuntimeError(f"공식 predict API와 다른 key: {key}")
    model = TabDPTRegressor(model_weight_path=str(pinned_weight), **CONSTRUCTOR)
    if model.max_features < len(FULL38):
        raise RuntimeError("고정 FULL38에서 PCA가 활성화됩니다. 명세 재검토 후 새 버전 필요")
    model.fit(X_train, y_train)
    return model


def predict_mean(model: Any, X_query: Any, seed: int,
                 *, execution_authorized: bool = False) -> Any:
    require_runtime(execution_authorized)
    import numpy as np
    q = np.asarray(X_query, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != len(FULL38) or np.isinf(q).any():
        raise ValueError("FULL38 순서의 질의 배열 필요")
    p = np.asarray(model.predict(q, seed=seed, **PREDICT), dtype=np.float64)
    if p.shape != (len(q),) or not np.isfinite(p).all():
        raise RuntimeError("예측 shape/유한성 실패")
    return p


def raw_query_audit(model: Any, X_query: Any, seed: int,
                    *, execution_authorized: bool = False) -> dict[str, Any]:
    """향후 첫 8 질의의 raw 단일/배치/순서/반복/다른 행 변화 감사.
    최종 shrink는 같은 farm-day의 과거 prefix를 유지하여 별도로 검사한다.
    """
    require_runtime(execution_authorized)
    import numpy as np
    q = np.asarray(X_query, dtype=np.float64)[:8].copy()
    if len(q) != 8:
        raise ValueError("감사에는 고정 8 질의가 필요")
    original = predict_mean(model, q, seed, execution_authorized=True)
    repeat = predict_mean(model, q, seed, execution_authorized=True)
    single = np.array([predict_mean(model, q[i:i+1], seed,
                                   execution_authorized=True)[0] for i in range(8)])
    reverse = predict_mean(model, q[::-1].copy(), seed, execution_authorized=True)[::-1]
    changed = q.copy()
    changed[1:] = np.where(np.isfinite(changed[1:]), changed[1:] + 1e4, 0.)
    other = predict_mean(model, changed, seed, execution_authorized=True)
    errors = {"repeat": float(np.max(np.abs(original-repeat))),
              "single": float(np.max(np.abs(original-single))),
              "reverse": float(np.max(np.abs(original-reverse))),
              "other_query": float(abs(original[0]-other[0]))}
    if any(error > RAW_ATOL for error in errors.values()):
        raise RuntimeError(f"raw 감사 실패(문턱 재조정 없음): {errors}")
    return {"status": "PASS", "raw_atol": RAW_ATOL, "max_abs": errors}


def combine_candidate(
    r3_raw: Any, tabdpt_raw: Any, query_frame: Any,
    shrink_fn: Callable[[Any, Any], Any], train_min: float, train_max: float,
    *, execution_authorized: bool = False,
) -> Any:
    """원 R3 raw + TabDPT raw를 먼저 혼합, 기존 shrink 1회, train bound clip 1회."""
    require_runtime(execution_authorized)
    import numpy as np
    a = np.asarray(r3_raw, dtype=np.float64)
    b = np.asarray(tabdpt_raw, dtype=np.float64)
    if a.shape != b.shape or a.shape != (len(query_frame),):
        raise ValueError("동일 query 순서의 raw component가 필요")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("raw prediction 유한성 실패")
    if not np.isfinite([train_min, train_max]).all() or train_min > train_max:
        raise ValueError("outer-train clip bound 실패")
    return np.clip(shrink_fn(.8*a + .2*b, query_frame), train_min, train_max)


if __name__ == "__main__":
    assert len(FULL38) == 38 and len(set(FULL38)) == 38
    assert FULL38[-1] == "season" and "day" not in FULL38
    print(json.dumps(SPEC, ensure_ascii=False, indent=2))

