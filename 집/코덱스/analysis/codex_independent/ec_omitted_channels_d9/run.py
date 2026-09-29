"""D9 label-free availability audit for raw channels absent from EC FULL."""
import env  # first project import

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / '온라인대회자료/정형데이터/참가자_배포/train_X.csv'
CHANNELS = ['in_rad', 'act_side', 'act_valve', 'act_cool', 'act_pump']


def main():
    x = pd.read_csv(SOURCE, usecols=['row_id', *CHANNELS])
    x = x[x.row_id.str[:3].isin(('F13', 'F47'))].copy()
    x['farm'] = x.row_id.str[:3]
    x['day'] = x.row_id.str[4:7].astype(int)
    assert len(x) == 9600 and x.groupby(['farm', 'day']).size().eq(24).all()
    result = {'protocol_sha256': hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256((HERE/'run.py').read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'channels': {}}
    for col in CHANNELS:
        result['channels'][col] = {}
        for farm, g in x.groupby('farm'):
            s = g[col]
            z = g.groupby('day')[col].agg(['min', 'max'])
            varying = z['min'].notna() & z['max'].notna() & z['min'].ne(z['max'])
            result['channels'][col][farm] = {
                'valid_rows': int(s.notna().sum()),
                'valid_fraction': float(s.notna().mean()),
                'nonzero_rows': int(s.notna().mul(s.ne(0)).sum()),
                'unique_valid_values': int(s.dropna().nunique()),
                'varying_days': int(varying.sum()),
                'varying_day_fraction': float(varying.mean()),
                'min': None if s.notna().sum() == 0 else float(s.min()),
                'max': None if s.notna().sum() == 0 else float(s.max()),
            }
    result['go_h18'] = any(
        all(result['channels'][c][f]['valid_fraction'] >= .1 and
            result['channels'][c][f]['nonzero_rows'] > 0 for f in ('F13', 'F47'))
        and any(result['channels'][c][f]['varying_day_fraction'] >= .05 for f in ('F13', 'F47'))
        for c in CHANNELS)
    out = ROOT / 'analysis/local/ec_omitted_channels_d9' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(out), **result}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
