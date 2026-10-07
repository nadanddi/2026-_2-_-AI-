"""Complete declared actuator-event responses before any domain model fit."""
from domain_features_v1 import build_domain as prior_build, dynamics, FAMILIES
from blk_baseline_data_v1 import np, pd
from fast_features_v2 import annotate

def event_response(observations, actuator, environment):
    """Change since last observed actuator value transition, using t-1 as origin.

    Missing hours/values discard the prior event; no effect is inferred at h0.
    This is an input co-change proxy, not a physiological causal-effect estimate.
    """
    h=observations.hour.to_numpy()
    av=observations[actuator].to_numpy(float)
    ev=np.asarray(environment,float)
    out=np.full(len(h),np.nan)
    before=None
    for i in range(len(h)):
        contiguous=i>0 and h[i]==h[i-1]+1
        valid=contiguous and np.isfinite(av[i]) and np.isfinite(av[i-1]) and np.isfinite(ev[i]) and np.isfinite(ev[i-1])
        if not valid:
            before=None
            continue
        if av[i]!=av[i-1]:before=ev[i-1]
        if before is not None:out[i]=ev[i]-before
    return out

def build_domain(frame):
    features,selected,metadata=prior_build(frame)
    z=annotate(frame)
    additions=[]
    for _,group in z.groupby(['farm','day'],sort=True):
        h=group.hour.to_numpy()
        index=group.row_id
        humidity_gap=features.loc[index,'humidity_gap'].to_numpy()
        vpd=features.loc[index,'vpd'].to_numpy()
        co2=group.in_co2.to_numpy(float)
        dose=group.act_co2.to_numpy(float)
        # Explicit onset/offset response: both hours and both measurements observed.
        onset=np.full(len(h),np.nan);offset=onset.copy()
        for i in range(1,len(h)):
            if h[i]!=h[i-1]+1 or not np.isfinite([dose[i],dose[i-1],co2[i],co2[i-1]]).all():continue
            if dose[i]>0 and dose[i-1]<=0:onset[i]=co2[i]-co2[i-1]
            if dose[i]<=0 and dose[i-1]>0:offset[i]=co2[i]-co2[i-1]
        event_series={
            'humidity_gap__vent_event_response':event_response(group,'act_vent',humidity_gap),
            'humidity_gap__fog_event_response':event_response(group,'act_fog',humidity_gap),
            'fog_vpd__event_response':event_response(group,'act_fog',vpd),
            'co2_dose_response__onset_delta':onset,
            'co2_dose_response__offset_delta':offset,
            'act_fog__temperature_response':event_response(group,'act_fog',group.in_temp.to_numpy(float)),
            'act_fog__humidity_response':event_response(group,'act_fog',group.in_hum.to_numpy(float)),
            'act_fog__vpd_response':event_response(group,'act_fog',vpd),
        }
        extra={}
        for name,values in event_series.items():
            extra[name]=values
            metadata[name]={'origin':name,'operator':'event-aligned observed input change','no_future':True,'not_causal_effect':True}
            for operator,observed in dynamics(pd.Series(values,index=h)).items():
                column=f'{name}__{operator}'
                extra[column]=observed
                metadata[column]={'origin':name,'operator':operator,'no_future':True}
        additions.append(pd.DataFrame(extra,index=index))
    features=pd.concat([features,pd.concat(additions)],axis=1)
    assert features.columns.is_unique and not np.isinf(features.to_numpy()).any()
    selected={candidate:[c for c in features if c==origin or c.startswith(origin+'__')] for candidate,origin in FAMILIES.items()}
    return features,selected,metadata
