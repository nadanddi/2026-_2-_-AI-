"""Registered domain families; same-record prefix only, no learned statistics."""
from blk_baseline_data_v1 import pd, np
from fast_features_v2 import build as base_build, annotate

WINDOWS = (1, 2, 3, 4, 6)
FAMILIES = {
    'D01':'vpd', 'D02':'rad_shade', 'D03':'rad_thermal', 'D04':'temp_gap',
    'D05':'humidity_gap', 'D06':'moisture_flux', 'D07':'fan_vpd',
    'D08':'heating_vpd', 'D09':'fog_vpd', 'D10':'vent_co2',
    'D11':'co2_dose_response', 'D12':'co2_uptake_proxy',
    'D13':'out_rad', 'D14':'in_temp', 'D15':'in_hum', 'D16':'in_co2',
    'D17':'act_vent', 'D18':'act_shade', 'D19':'act_thermal',
    'D20':'act_heating', 'D21':'act_circfan', 'D22':'act_fog',
    'D23':'act_co2', 'D24':'sealed_run',
}

def dynamics(series):
    h = series.index.to_numpy()
    a = series.to_numpy(float)
    columns = {}
    for lag in WINDOWS:
        past = series.reindex(h-lag).to_numpy()
        columns[f'lag{lag}'] = past
        columns[f'rate{lag}'] = (a-past)/lag
    columns['d2'] = a-2*series.reindex(h-1).to_numpy()+series.reindex(h-2).to_numpy()
    for window in WINDOWS:
        dense=series.reindex(np.arange(24)).to_numpy()
        matrix=np.lib.stride_tricks.sliding_window_view(np.pad(dense,(window-1,0),constant_values=np.nan),window)[h]
        finite=np.isfinite(matrix)
        counts=finite.sum(axis=1)
        sums=np.where(finite,matrix,0).sum(axis=1)
        means=np.divide(sums,counts,out=np.full(len(h),np.nan),where=counts>0)
        sq=np.where(finite,(matrix-means[:,None])**2,0).sum(axis=1)
        std=np.sqrt(np.divide(sq,counts,out=np.full(len(h),np.nan),where=counts>0))
        for op,value in [('mean',means),('std',std),('sum',np.where(counts>0,sums,np.nan)),('count',counts)]:
            columns[f'{op}{window}']=value
    return columns

def build_domain(frame):
    features, metadata=base_build(frame)
    annotated=annotate(frame)
    additions=[]
    for _,group in annotated.groupby(['farm','day'],sort=True):
        extra={}
        hours=group.hour.to_numpy()
        for origin in ['co2_dose_response','co2_uptake_proxy','sealed_run']:
            series=pd.Series(features.loc[group.row_id,origin].to_numpy(),index=hours)
            for operator,values in dynamics(series).items():
                name=f'{origin}__{operator}'
                extra[name]=values
                metadata[name]={'origin':origin,'operator':operator,'prefix_only':True}
        additions.append(pd.DataFrame(extra,index=group.row_id))
    features=pd.concat([features,pd.concat(additions)],axis=1)
    assert features.index.is_unique and features.columns.is_unique
    assert not np.isinf(features.to_numpy()).any()
    selected={candidate:[column for column in features if column==origin or column.startswith(origin+'__')]
              for candidate,origin in FAMILIES.items()}
    assert len(selected)==24 and all(selected.values())
    return features, selected, metadata
