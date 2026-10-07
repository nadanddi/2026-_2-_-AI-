from pathlib import Path
import json
from ch2_reference_loader_v1 import load_reference
from ch2_query_context_v2 import CH2QueryContext
HERE=Path(__file__).resolve().parent
model,layout,_=load_reference();ctx=CH2QueryContext(layout,model)
pred=json.loads((HERE/'checkpoints/CH2_BLK_v1/predictions.json').read_text(encoding='utf-8'))
result=None
for i,rid in enumerate(pred['row_ids']):
    packet=ctx.query_prefix(rid)
    for method in pred['diagnostics']:
        _,diag=model.choose(rid,packet,method);stored=pred['diagnostics'][method][i]
        if diag!=stored:
            normalized=json.loads(json.dumps(diag,allow_nan=False))
            assert normalized==stored, 'failure is not explained by JSON tuple serialization'
            result={'status':'FIRST_GATE_DIAG_MISMATCH_IS_JSON_TUPLE_LIST_ONLY','row_index':i,'row_id':rid,'method':method,
                    'differing_value_types':{k:[type(v).__name__,type(stored[k]).__name__] for k,v in diag.items() if v!=stored[k]},
                    'JSON_normalized_exact':True,'predictions_changed':False,'heldout_truth_loaded':False}
            break
    if result:break
assert result
out=HERE/'CH2_gate_serialization_diagnosis_v1.json';assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
