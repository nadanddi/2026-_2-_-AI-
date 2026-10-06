from pathlib import Path
import sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H));import runner_v4 as R
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    D=H/'ablation_score_v1';groups=pd.read_csv(D/'groups.csv',dtype={'seed':str});cases=pd.read_csv(D/'cases.csv',dtype={'seed':str});support=pd.read_csv(H/'influence_support_profiles_v1.csv')
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
    fig,axs=plt.subplots(1,3,figsize=(15,4.4));colors=['#55616e','#d98824','#307c9a'];arms=['BASE','D1','D2'];labels=['현재 EC14','ET: 139일 제외','ET: 139·231일 제외']
    for i,(arm,color) in enumerate(zip(arms,colors)):
        c=cases[(cases.arm==arm)&cases.seed.eq('ensemble')&cases.farm.eq('F47')&cases.day.eq(161)].iloc[0]
        axs[0].bar(i,c.prediction,color=color);axs[0].text(i,c.prediction+.02,f'{c.prediction:.3f}',ha='center');truth=c.truth
    axs[0].axhline(truth,color='#ac3434',ls='--',label=f'실제 EC {truth:.3f}');axs[0].set_xticks(range(3),['현재','D1','D2']);axs[0].set_ylabel('24시간 평균 EC');axs[0].set_title('F47 161일 최종 예측');axs[0].legend(loc='lower left',fontsize=9)
    strata=['ordinary_closed61','ordinary_other268','high'];xs=np.arange(3)
    for i,(arm,color,label) in enumerate(zip(arms,colors,labels)):
        g=groups[groups.arm.eq(arm)&groups.seed.eq('ensemble')].set_index('group');ys=[g.loc[s,'rmse'] for s in strata];axs[1].bar(xs+(i-1)*.25,ys,width=.24,color=color,label=label)
    axs[1].set_xticks(xs,['일반·환기 0 비율\n80% 이상 (61일)','그 외 일반\n(268일)','고EC\n(31일)']);axs[1].set_ylabel('시간행 RMSE');axs[1].set_title('특정 날의 개선과 전체 손실');axs[1].legend(fontsize=8)
    for i,(arm,color) in enumerate(zip(arms,colors)):
        p=support[(support.arm==arm)&support.level.eq('smooth_day')].iloc[0];axs[2].bar(i,100*p.high_day_weight,color=color);axs[2].text(i,100*p.high_day_weight+1,f'{100*p.high_day_weight:.1f}%',ha='center')
    axs[2].set_xticks(range(3),['현재','D1','D2']);axs[2].set_ylim(0,100);axs[2].set_ylabel('고EC 학습날 지원 가중치 (%)');axs[2].set_title('ET 시드7: 평활 평균의 학습지원')
    for ax in axs:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('공개 DIAG10 360일 · ET만 재학습한 조건부 진단',fontsize=14);fig.tight_layout();p=H/'진단비교_v2.png';assert not p.exists();fig.savefig(p,dpi=160,bbox_inches='tight');print('PLOT_COMPLETE',flush=True)
if __name__=='__main__':main()

