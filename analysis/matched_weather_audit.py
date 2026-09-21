"""Training-only descriptive matched-weather audit; no chronology reconstruction."""
import json
import numpy as np
import pandas as pd
from profile_tabular import load, OUT, TARGETS


def main():
    x = load('train_X.csv')
    y = load('train_y.csv')
    data = x.merge(y[['row_id'] + TARGETS], on='row_id', how='left', validate='one_to_one')
    result = {}
    columns = ['in_temp', 'in_hum', 'in_co2', 'act_circfan', 'act_vent',
               'act_heating', 'act_co2', 'act_fog', 'act_shade', 'act_thermal'] + TARGETS
    for farm in ['F13', 'F47']:
        g = data[data.farm.eq(farm)].sort_values('time').copy()
        days = pd.read_csv(OUT / f'house_daily_{farm}.csv', dtype={'signature': str})
        daily = g.groupby('day')[columns].mean()
        records = []
        for signature, group in days.groupby('signature'):
            if len(group) != 2:
                continue
            a, b = sorted(group.day.astype(int))
            weather = ['out_temp', 'out_hum', 'out_rad', 'out_wspd']
            wa = g[g.day.eq(a)].sort_values('hour')[weather].to_numpy()
            wb = g[g.day.eq(b)].sort_values('hour')[weather].to_numpy()
            assert wa.shape == (24, 4) and np.isfinite(wa).all()
            assert np.array_equal(wa, wb), 'Cached weather match no longer agrees with raw inputs'
            row = dict(first_day=a, second_day=b, gap=b-a, period=(a//30)*30)
            for c in columns:
                row[c + '_delta'] = daily.loc[b, c] - daily.loc[a, c]
            records.append(row)
        pairs = pd.DataFrame(records)
        pairs.to_csv(OUT / f'matched_weather_pairs_{farm}.csv', index=False)
        adjacent = pairs[pairs.gap.eq(1)].copy()
        associations = []
        for period, p in [('all', adjacent)] + [(str(k), v) for k, v in adjacent.groupby('period')]:
            for c in columns[:-2]:
                valid = p[[c+'_delta', 'sub_ec_delta']].dropna()
                nonzero = valid[(valid[c+'_delta'].abs()>1e-9) & (valid.sub_ec_delta.abs()>1e-9)]
                associations.append(dict(period=period, variable=c, n=len(valid),
                    pearson=valid[c+'_delta'].corr(valid.sub_ec_delta) if len(valid)>2 and valid[c+'_delta'].std()>0 and valid.sub_ec_delta.std()>0 else None,
                    nonzero_n=len(nonzero), opposite_sign_fraction=float((nonzero[c+'_delta']*nonzero.sub_ec_delta<0).mean())))
        pd.DataFrame(associations).to_csv(OUT / f'matched_weather_associations_{farm}.csv', index=False)
        continuity = pd.read_csv(OUT / f'house_continuity_{farm}.csv')
        ec = continuity[continuity.column.eq('sub_ec')].copy()
        ec['same_slot_advantage'] = ec.nominal_abs_diff - ec.same_slot_abs_diff
        indexed = g.set_index(['day', 'hour']).sub_ec
        for row in ec.itertuples():
            current = indexed.loc[(row.current_day, 0)]
            assert np.isclose(row.same_slot_abs_diff, abs(current-indexed.loc[(row.current_day-2, 23)]))
            assert np.isclose(row.nominal_abs_diff, abs(current-indexed.loc[(row.current_day-1, 23)]))
        losses = ec[ec.same_slot_advantage.le(0)].sort_values('same_slot_advantage')
        losses.to_csv(OUT / f'matched_weather_continuity_exceptions_{farm}.csv', index=False)
        # Official consecutive hours only; never bridge missing dates.
        g['ec_jump'] = g.sub_ec.diff().abs().where(g.time.diff().eq(1))
        sig = days.set_index('day').signature
        midnight = g[g.hour.eq(0) & g.ec_jump.notna()].copy()
        midnight['weather_repeat'] = midnight.day.map(sig).eq((midnight.day-1).map(sig))
        boundary = []
        for repeated, z in midnight.groupby('weather_repeat'):
            boundary.append(dict(same_weather_as_previous_day=bool(repeated), n=len(z),
                large_ec_jump_n=int(z.ec_jump.gt(.2).sum()), mean_abs_ec_jump=float(z.ec_jump.mean())))
        result[farm] = dict(adjacent_pairs=len(adjacent), nonadjacent_pairs=int(pairs.gap.ne(1).sum()),
            ec_second_higher=int(adjacent.sub_ec_delta.gt(0).sum()),
            ec_second_lower=int(adjacent.sub_ec_delta.lt(0).sum()),
            ec_second_lower_by_over_point1=adjacent.loc[adjacent.sub_ec_delta.lt(-.1), ['first_day','second_day','sub_ec_delta','act_circfan_delta']].to_dict('records'),
            pooled_associations=[r for r in associations if r['period']=='all'],
            continuity_endpoints=len(ec), continuity_losses=losses.to_dict('records'),
            midnight=boundary,
            nonadjacent= pairs.loc[pairs.gap.ne(1), ['first_day','second_day','gap','sub_ec_delta','in_temp_delta']].to_dict('records'))
    def clean(v):
        if isinstance(v, dict): return {k:clean(z) for k,z in v.items()}
        if isinstance(v, list): return [clean(z) for z in v]
        if isinstance(v, (float, np.floating)) and not np.isfinite(v): return None
        if isinstance(v, np.generic): return v.item()
        return v
    result = clean(result)
    (OUT / 'matched_weather_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
