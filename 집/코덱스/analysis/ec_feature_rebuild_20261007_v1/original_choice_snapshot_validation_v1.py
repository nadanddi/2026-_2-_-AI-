"""Stored choice metadata contract; not fresh selection or causal proof."""
import hashlib
import json
import math


def validate(snapshot, ids, expected_sources):
    if type(snapshot) is not dict: raise ValueError('choice snapshot type')
    expected_fields = {'status','row_ids','choices','choices_sha256','sources_sha256','scope',
                       'source_checks','source_maximum_difference','all_choices_present',
                       'heldout_truth_loaded','whole_pipeline_gate_passed'}
    if set(snapshot) != expected_fields: raise ValueError('choice snapshot fields')
    if snapshot['status'] != 'ORIGINAL_SG2_CHOICES_NOT_FULL_MODEL_GATE' or snapshot['scope'] != 'BLK_RAW_PASS':
        raise ValueError('choice snapshot policy')
    if snapshot['all_choices_present'] is not True or snapshot['heldout_truth_loaded'] is not False or snapshot['whole_pipeline_gate_passed'] is not False:
        raise ValueError('choice snapshot flags')
    if snapshot['row_ids'] != ids or type(snapshot['choices']) is not dict or set(snapshot['choices']) != set(ids):
        raise ValueError('choice snapshot row coverage')
    if type(snapshot['source_checks']) is not int or snapshot['source_checks'] != 3*len(ids):
        raise ValueError('choice source replay count')
    value = snapshot['source_maximum_difference']
    if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 1e-12:
        raise ValueError('choice source replay difference')
    if snapshot['sources_sha256'] != expected_sources or len(expected_sources) != 5:
        raise ValueError('choice source binding')
    raw = json.dumps(snapshot['choices'],sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    if hashlib.sha256(raw).hexdigest() != snapshot['choices_sha256']:
        raise ValueError('choice payload digest')
    return snapshot['choices']
