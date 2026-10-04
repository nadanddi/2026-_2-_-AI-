"""샌드박스 metadata 실패 진단만. 버전 수정·타사 import 없음."""
from pathlib import Path
import importlib.metadata as md
import json
import sys
ROOT = Path(__file__).resolve().parent.parents[3]
SITE = ROOT / "집" / "코덱스" / "local" / "tabdpt130_cpu_v1" / "site"
sys.path[:0] = [str(SITE), str(ROOT / ".analysis-tools" / "python"),
               str(ROOT / ".analysis-tools" / "extra")]
items = []
for name in ("tabdpt", "faiss-cpu", "omegaconf",
             "antlr4-python3-runtime", "huggingface-hub"):
    try:
        d = md.distribution(name)
        item = {"name": name, "metadata_version": d.version,
                "selected_path": str(d._path)}
        try:
            text = (Path(d._path) / "METADATA").read_text(encoding="utf-8")
            item["direct_metadata_read"] = "PASS"
        except OSError as e:
            item.update(direct_metadata_read="FAIL", error_type=type(e).__name__,
                        error=str(e))
    except Exception as e:
        item={"name": name, "error_type": type(e).__name__, "error": str(e)}
    items.append(item)
print(json.dumps({"python": sys.version, "executable": sys.executable,
                  "results": items, "third_party_imports": 0, "model_fit": 0,
                  "weight_reads": 0, "data_reads": 0}, ensure_ascii=False, indent=2))

