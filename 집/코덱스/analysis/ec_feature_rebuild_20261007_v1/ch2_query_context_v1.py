"""Lazy same-block query_prefix adapter; future/gap values never converted."""
from pathlib import Path
import csv,math
from blk_context_v1 import BLKContext,RAW,DATA,key

class CH2QueryContext(BLKContext):
    def __init__(self,layout,model):
        # Retain the established query_prefix boundary. References come from the audited file.
        self.train_ids=set(layout['train_ids']);self.query_ids=set(layout['query_ids'])
        self.gap_ids=set(layout['gap_ids_REMOVE_INPUT_AND_BOTH_LABELS'])
        assert self.query_ids==set(model.query_ids)
        self.query_to_block=model.query_to_block;self.blocks=model.blocks
        self.reference_inputs={};self.reference_labels={};self._query={}
        self.consumed=[]
        with (DATA/'train_X.csv').open(encoding='utf-8-sig',newline='') as handle:
            for row in csv.DictReader(handle):
                rid=row['row_id']
                if rid not in self.query_ids:continue
                assert rid not in self._query
                # CSV strings only: no float, pivot, statistics, labels or fitted transformation.
                self._query[rid]={c:row[c] for c in RAW}
        assert set(self._query)==self.query_ids
        assert not(self.train_ids&self.query_ids or self.query_ids&self.gap_ids or self.train_ids&self.gap_ids)

    def query_prefix(self,rid):
        assert rid in self.query_ids
        farm,day,hour=key(rid);index=self.query_to_block[rid]
        expected={f'{farm}_{d:03d}_{h:02d}' for d in self.blocks[index]['query_days'] if d<=day
                  for h in range(24 if d<day else hour+1)}
        values={}
        for other in sorted(expected):
            assert other in self._query and self.query_to_block[other]==index
            assert key(other)[0]==farm and key(other)[1:]<=(day,hour)
            raw=self._query[other]
            assert set(raw)==set(RAW)
            obs={c:float(raw[c]) if raw[c].strip() else None for c in RAW}
            assert all(v is None or math.isfinite(v) for v in obs.values())
            values[other]=obs
        self.consumed.append({'rid':rid,'ordered_ids':sorted(expected)})
        return values
