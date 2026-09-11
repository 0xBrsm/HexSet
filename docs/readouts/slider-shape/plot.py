"""Standalone shape and paired-strength figures from independently audited JSON."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REGIMES=('none','moderate','heavy','full','mixed','rising','falling')
COLORS={'square':'#245985','linear':'#555555','sqrt':'#b95826'}


def save(fig,path,name):
    fig.savefig(path/f'{name}.png',dpi=160)
    fig.savefig(path/f'{name}.svg')
    plt.close(fig)


def plot(screen,path,confirmation=None):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(6,4.5))
    x=np.linspace(0,1,201)
    for name,y in [('square',x*x),('linear',x),('sqrt',np.sqrt(x))]:
        ax.plot(x,y,label=name,color=COLORS[name],linewidth=2)
    ax.set(xlabel='Observed exchange activity (last eight eligible turns)',
           ylabel='Slider position: 0 = N, 1 = M',xlim=(0,1),ylim=(0,1),
           title='Only the mapping changes; endpoints and memory are fixed')
    ax.legend(frameon=False);ax.grid(alpha=.15);fig.tight_layout();save(fig,path,'shape-mappings')
    for data in [screen]+([confirmation] if confirmation else []):
        fig,ax=plt.subplots(figsize=(10,6))
        for offset,(c,primary) in enumerate(data['primary'].items()):
            rows=[next(r for r in data['per_regime'] if r['regime']==g and r['candidate']==c) for g in REGIMES]+[primary]
            y=np.arange(8)+(offset-.5)*.16 if len(data['primary'])==2 else np.arange(8)
            d=np.array([100*r['difference'] for r in rows])
            lo=np.array([100*r['interval95'][0] for r in rows]);hi=np.array([100*r['interval95'][1] for r in rows])
            ax.errorbar(d,y,xerr=[d-lo,hi-d],fmt='o',color=COLORS[c],capsize=3,label=f'{c} minus linear')
        ax.set_yticks(range(8),[g.title() for g in REGIMES]+['Equal-condition aggregate'])
        ax.invert_yaxis();ax.axvline(0,color='gray',linewidth=1)
        ax.axvline(1,color='#555555',linestyle=':',label='+1 pp meaningful-improvement margin')
        ax.set_xlabel('Difference in focal win rate (percentage points)')
        ax.set_title(f"{data['stage'].title()}: paired nonlinear-minus-linear comparisons")
        ax.grid(axis='x',alpha=.15);ax.legend(frameon=False)
        note='Two-sided 95% intervals. Screen comparisons are descriptive and select at most one challenger.'
        if data['stage']=='confirmation':
            d=next(iter(data['primary'].values()))
            note=f"Two-sided 95% intervals; registered aggregate one-sided 95% lower bound: {100*d['lower95']:+.2f} pp."
        fig.text(.5,.01,note,ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.035,1,1));save(fig,path,f"{data['stage']}-comparisons")
    data=confirmation or screen
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,g in zip(axes,('full','rising','falling')):
        for d in data['diagnostics']:
            if d['regime']!=g:continue
            bins=['0-19','20-39','40-59','60-79','80-99']
            ax.plot([10,30,50,70,90],[d['bins'][b]['mean_alpha'] for b in bins],
                    'o-',color=COLORS[d['candidate']],label=d['candidate'])
        ax.set_title(g.title());ax.set_xlabel('Global turn (20-turn bins)')
        ax.axvline(40,color='gray',linestyle=':',alpha=.6);ax.grid(alpha=.15)
    axes[0].set_ylabel('Mean applied slider position');axes[0].set_ylim(0,1);axes[-1].legend(frameon=False)
    fig.suptitle(f"Applied response ({data['stage']}): means among games reaching each bin")
    fig.tight_layout();save(fig,path,'activity-response')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--confirmation',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plot(json.loads(a.screen.read_text()),a.out,json.loads(a.confirmation.read_text()) if a.confirmation else None)
