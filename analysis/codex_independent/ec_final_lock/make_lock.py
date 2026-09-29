"""Choose a new final-check day set from row identifiers alone."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'온라인대회자료/정형데이터/참가자_배포'
R=ROOT/'analysis/local'
SEARCH=R/'ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM=R/'ec_locked_confirmation/20260928_044934'


def main():
    ids=pd.read_csv(DATA/'train_X.csv',usecols=['row_id']).row_id
    train={(s[:3],int(s[4:7])) for s in ids if s[:3] in ('F13','F47')}
    assert len(train)==400
    paths=[*[SEARCH/f'fold{i}.csv' for i in (0,2,4,6)],
           CONFIRM/'fold8.csv',CONFIRM/'fold9.csv']
    seen=set()
    for path in paths:
        for s in pd.read_csv(path,usecols=['row_id']).row_id:
            seen.add((s[:3],int(s[4:7])))
    assert len(seen)==234 and seen<=train
    pool=train-seen
    assert len(pool)==166
    chosen=[]
    for farm in ('F13','F47'):
        for section,n in (('first',15),('second',5)):
            items=[(f,d) for f,d in pool if f==farm and (d<179)==(section=='first')]
            assert len(items)>=n
            items.sort(key=lambda item:hashlib.sha256(
                f'ec-final-lock-20260929|{item[0]}|{item[1]}'.encode()).hexdigest())
            chosen.extend(items[:n])
    chosen=sorted(chosen)
    assert len(chosen)==40 and len(set(chosen))==40
    result=dict(salt='ec-final-lock-20260929',source='train_X.row_id and prior OOF row_id only',
                eligible_days=166,selected_days=40,
                selected=[dict(farm=f,day=d,section='first' if d<179 else 'second')
                          for f,d in chosen],
                no_labels_read=True,test_input_read=False,
                protocol_sha256=hashlib.sha256(Path(__file__).with_name('PROTOCOL.md').read_bytes()).hexdigest(),
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out=Path(__file__).with_name('locked_days.json')
    assert not out.exists()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='selected'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
