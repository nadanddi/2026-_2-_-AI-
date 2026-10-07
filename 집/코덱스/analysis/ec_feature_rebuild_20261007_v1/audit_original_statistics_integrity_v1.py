"""Synthetic malformed-draw failures using tiny isolated files, no target values."""
from pathlib import Path
import json,hashlib,tempfile,struct,copy
import original_statistics_integrity_v1 as guard
HERE=Path(__file__).resolve().parent
layout={'block_keys':[['F13',0],['F47',0]],'groups':[[0],[1]],
        'rows_per_block':[24,24],'TM_rows_per_block':[24,24]}
valid=struct.pack('<2H',1,1)*4;cases=[]
with tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic_stat_integrity_') as temp:
    path=Path(temp)/'draws.bin'
    def invoke(raw,design=layout,expected=None):
        path.write_bytes(raw)
        return guard.validate_draw_file(path,expected or guard.sha(path),design,4)
    assert invoke(valid)==4;cases.append({'case':'valid_tiny_draws','PASS':True})
    variants=[('wrong_SHA',valid,layout,'0'*64),
              ('truncated',valid[:-1],layout,None),('extra_byte',valid+b'x',layout,None),
              ('wrong_farm_total',struct.pack('<2H',2,0)*4,layout,None)]
    bad=copy.deepcopy(layout);bad['rows_per_block']=[24];variants.append(('layout_width',valid,bad,None))
    bad=copy.deepcopy(layout);bad['groups']=[[0],[0]];variants.append(('duplicate_group',valid,bad,None))
    bad=copy.deepcopy(layout);bad['groups']=[[0],[]];variants.append(('empty_group',valid,bad,None))
    bad=copy.deepcopy(layout);bad['TM_rows_per_block']=[0,0];variants.append(('zero_TM_denominator',valid,bad,None))
    bad=copy.deepcopy(layout);bad['TM_rows_per_block']=[25,24];variants.append(('TM_exceeds_rows',valid,bad,None))
    for name,raw,design,expected in variants:
        try:invoke(raw,design,expected)
        except AssertionError:cases.append({'case':name+'_rejected','PASS':True})
        else:raise AssertionError(name+' accepted')
result={'status':'SYNTHETIC_DRAW_INTEGRITY_NEGATIVE_PASS','cases':cases,
        'code_sha256':guard.sha(__file__),'guard_sha256':guard.sha(guard.__file__),
        'model_fit':False,'heldout_truth_loaded':False,'full_model_score_gate':False}
with (HERE/'DOMAIN24_original_statistics_integrity_synthetic_v1.json').open('x',encoding='utf-8') as h:
    json.dump(result,h,ensure_ascii=False,indent=2)
print(f'Synthetic malformed-draw {len(cases)} checks PASS; no model/truth')
