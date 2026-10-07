"""v2 lineage plus the exact method context/layout pin before integration."""
from pathlib import Path
import json
from blk_assemble_predictions_v2 import preflight,_sha
HERE=Path(__file__).resolve().parent

if __name__=='__main__':
    lineage=preflight()
    rules=json.loads((HERE/'BLK_method_registration_v3.json').read_text(encoding='utf-8'))
    assert rules['layout_sha256']==_sha(HERE/'BLK_layout_v2.json')
    assert rules['context_code_sha256']==_sha(HERE/'blk_context_v1.py')
    lineage['assembler_sources_sha256']['blk_assemble_predictions_v3.py']=_sha(__file__)
    code=(HERE/'blk_assemble_predictions_v1.py').read_text(encoding='utf-8')
    code=code.replace('checkpoints/BLK_ASSEMBLED_v1','checkpoints/BLK_ASSEMBLED_v2')
    code=code.replace("manifest={'status':", "manifest={'lineage':lineage,'status':")
    exec(compile(code,str(Path(__file__).resolve()),'exec'),globals())
