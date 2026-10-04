"""표준 라이브러리 메타데이터 읽기 전용. 패키지 import/설치/모델 fit 없음."""
from pathlib import Path
import importlib.metadata as metadata
import json
import platform
import sys

ROOT = Path(__file__).resolve().parent.parents[3]
SEARCH_PATHS = [ROOT / ".analysis-tools" / "python", ROOT / ".analysis-tools" / "extra"]
NAMES = {"torch", "numpy", "scipy", "scikit-learn", "faiss-cpu",
         "huggingface-hub", "omegaconf", "safetensors", "tqdm", "tabdpt"}

def normalize(value):
    return value.lower().replace("_", "-")

def inspect_metadata():
    rows = []
    effective = {}
    for location in SEARCH_PATHS:
        for dist in metadata.distributions(path=[str(location)]):
            name = normalize(dist.metadata.get("Name", ""))
            if name in NAMES:
                row = dict(name=name, version=dist.version,
                           metadata_path=str(dist._path), search_path=str(location))
                rows.append(row)
                effective.setdefault(name, dist.version)
    return dict(scope="read-only distribution metadata; no third-party imports",
                python=sys.version, executable=sys.executable,
                platform=platform.platform(), machine=platform.machine(),
                search_paths=list(map(str, SEARCH_PATHS)), distributions=rows,
                effective_versions=effective,
                absent=sorted(NAMES-set(effective)),
                imports_of_candidate_packages=0, installs=0, fits=0)

if __name__ == "__main__":
    print(json.dumps(inspect_metadata(), ensure_ascii=False, indent=2))

