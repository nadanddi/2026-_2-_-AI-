"""Training data only figures. Saved locally, not published."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.analysis-tools/python'))
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from profile_tabular import load, OUT, TARGETS


def main():
    x=load('train_X.csv'); y=load('train_y.csv')
    d=x.merge(y[['row_id']+TARGETS],on='row_id',how='left')
    colors={'F13':'#16618a','F47':'#c7612c'}
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for f in colors:
        g=d[d.farm==f].sort_values('time')
        daily=g.groupby('day').sub_ec.mean()
        daily=daily.reindex(range(int(daily.index.min()),int(daily.index.max())+1))
        axes[0,0].plot(daily.index,daily.values,'.-',lw=.7,ms=2,label=f,color=colors[f])
        delta=g.sub_ec.diff().abs().where(g.time.diff().eq(1))
        byhour=delta.groupby(g.hour).mean()
        axes[0,1].plot(byhour.index,byhour.values,'o-',ms=3,label=f,color=colors[f])
        gg=g.set_index('time')
        values=[]
        for lag in range(13):
            older=gg.in_temp.reindex(gg.index-lag).set_axis(gg.index)
            values.append(older.corr(gg.sub_temp))
        axes[1,0].plot(range(13),values,'o-',ms=3,label=f,color=colors[f])
    g=d[d.farm=='F13']
    for day,style in [(10,'-'),(11,'--')]:
        z=g[g.day==day].sort_values('hour')
        axes[1,1].plot(z.hour,z.out_temp,style,lw=2,label=f'out_temp day {day}')
        axes[1,1].plot(z.hour,z.in_temp,style,lw=1,label=f'in_temp day {day}',alpha=.65)
    axes[0,0].set(title='EC daily means: slow periods and isolated high days',xlabel='Farm-relative day',ylabel='EC (dS/m)')
    axes[0,1].set(title='EC changes concentrate at midnight',xlabel='Hour at end of one-hour difference',ylabel='Mean absolute EC difference (dS/m)')
    axes[1,0].set(title='Past indoor temperature better tracks substrate temperature',xlabel='Input lag (hours)',ylabel='Pearson correlation')
    axes[1,1].set(title='F13: repeated external weather, different indoor temperature',xlabel='Hour',ylabel='Temperature (C)')
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Tabular mission: training-data evidence',fontsize=16)
    fig.savefig(OUT/'training_patterns.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
