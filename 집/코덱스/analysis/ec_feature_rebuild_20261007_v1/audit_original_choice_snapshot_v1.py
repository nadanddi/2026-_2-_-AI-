"""Synthetic metadata mutation refusals; no model or dataset access."""
from pathlib import Path
import copy
import hashlib
import json
import original_choice_snapshot_validation_v1 as guard

HERE = Path(__file__).resolve().parent


def main():
    pins = {str((HERE/n).resolve()):hashlib.sha256((HERE/n).read_bytes()).hexdigest()
            for n in ('audit_original_choice_snapshot_v1.py','original_choice_snapshot_validation_v1.py')}
    ids = ['F13_179_00','F47_179_00']
    choices = {rid:{'active':True,'has_candidate':False,'reference_day':None,'level':None} for rid in ids}
    sources = {f'synthetic_source_{i}':'a'*64 for i in range(5)}
    snap = {'status':'ORIGINAL_SG2_CHOICES_NOT_FULL_MODEL_GATE','row_ids':ids,'choices':choices,
            'choices_sha256':hashlib.sha256(json.dumps(choices,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
            'sources_sha256':sources,'scope':'BLK_RAW_PASS','source_checks':6,
            'source_maximum_difference':0.,'all_choices_present':True,'heldout_truth_loaded':False,
            'whole_pipeline_gate_passed':False}
    assert guard.validate(snap,ids,sources) == choices
    invalid = [('status','PASS'),('scope','BLK_QUERY_ROLE'),('all_choices_present',False),
               ('heldout_truth_loaded',True),('whole_pipeline_gate_passed',True),
               ('source_checks',True),('source_checks',5),('source_maximum_difference',-1.),
               ('source_maximum_difference',float('nan')),('source_maximum_difference',1e-8),
               ('choices_sha256','b'*64),('sources_sha256',{}),('row_ids',list(reversed(ids))),
               ('choices',{}),('all_choices_present',1),('heldout_truth_loaded',0)]
    refused = []
    for key,value in invalid:
        modified = copy.deepcopy(snap); modified[key] = value
        try: guard.validate(modified,ids,sources)
        except ValueError: refused.append(key+':'+repr(value))
        else: raise AssertionError('bad snapshot accepted '+key)
    for change in ('extra','missing'):
        modified = copy.deepcopy(snap)
        if change == 'extra': modified['extra'] = True
        else: del modified['scope']
        try: guard.validate(modified,ids,sources)
        except ValueError: refused.append(change+' field')
        else: raise AssertionError('wrong fields accepted')
    changed = copy.deepcopy(snap); changed['choices'][ids[0]]['active'] = False
    try: guard.validate(changed,ids,sources)
    except ValueError: refused.append('choice payload mutation')
    else: raise AssertionError('changed payload accepted')
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path,value in pins.items())
    result = {'status':'SYNTHETIC_CHOICE_METADATA_PASS_NOT_SELECTION_OR_FULL_GATE',
              'positive_checks':1,'invalid_checks':refused,'invalid_count':len(refused),
              'executed_sources_sha256':pins,'model_fit':False,'heldout_truth_read':False,
              'whole_pipeline_gate_passed':False,'limits':['Metadata only; no fresh reference-selection or causal proof']}
    with (HERE/'original_choice_snapshot_synthetic_audit_v1.json').open('x',encoding='utf-8') as out:
        json.dump(result,out,ensure_ascii=False,indent=2)
    print('Choice metadata synthetic PASS',len(refused),'refusals')


if __name__ == '__main__': main()
