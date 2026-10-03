from prepare import *
import os
os.environ['MPLCONFIGDIR']=str(O/'mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf');plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
def main():
 raw=pd.read_csv(O/'temperature_raw_public.csv');fig,axes=plt.subplots(1,2,figsize=(12,4.3),sharey=True)
 for ax,(farm,day) in zip(axes,[('F13',194),('F47',191)]):
  q=raw[(raw.farm==farm)&(raw.day==day)].sort_values('hour');ax.plot(q.hour,q.in_temp,color='#94a3b8',linestyle='--',label='실내 온도');ax.plot(q.hour,q.sub_temp,color='#0f766e',linewidth=2.5,label='실제 배지');ax.plot(q.hour,q.prediction,color='#dc2626',linewidth=2,label='예측 배지');ax.fill_between(q.hour,q.sub_temp,q.prediction,color='#dc2626',alpha=.12);ax.set_title(f'{farm}/{day} · '+('성공' if farm=='F13' else '실패'));ax.set_xlabel('시각');ax.set_xticks([0,6,12,18,23]);ax.grid(alpha=.2);ax.legend(fontsize=9)
 axes[0].set_ylabel('온도(℃)');fig.suptitle('같은 외부 날씨, 다른 배지 온도 수준');fig.tight_layout();fig.savefig(H/'두날_예측경로.png',dpi=170);plt.close(fig)
if __name__=='__main__':main()
