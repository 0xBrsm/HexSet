"""Standalone descriptive curves and public-activity trajectories."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REGIMES=('none','moderate','heavy','full','mixed','rising','falling')
FIXED=('0','.25','.5','.75','1')


def plot(screen,path,confirmation=None):
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,4,figsize=(15,8),sharex=True,sharey=True)
 for ax,regime in zip(axes.flat,REGIMES):
  rows={r['candidate']:r for r in screen['rows'] if r['regime']==regime}
  x=[float(p) for p in FIXED];y=[100*rows[p]['rate'] for p in FIXED]
  errors=[[100*(rows[p]['rate']-rows[p]['wilson95'][0]) for p in FIXED],
          [100*(rows[p]['wilson95'][1]-rows[p]['rate']) for p in FIXED]]
  ax.errorbar(x,y,yerr=errors,fmt='o-',color='#245985',capsize=3,label='Fixed slider positions')
  a=rows['A'];ax.axhline(100*a['rate'],color='#b95826',linestyle='--',label='Adaptive policy')
  ax.axhspan(100*a['wilson95'][0],100*a['wilson95'][1],color='#b95826',alpha=.10)
  ax.set_title(regime.title());ax.set_xticks(x);ax.set_ylim(0,65);ax.grid(alpha=.15)
  ax.set_xlabel('Fixed position: 0 = N, 1 = M');ax.set_ylabel('Focal win rate (%)')
 axes.flat[-1].axis('off')
 handles,labels=axes.flat[0].get_legend_handles_labels()
 axes.flat[-1].legend(handles,labels,loc='upper left',frameon=False)
 axes.flat[-1].text(0,.62,'64 matched games per policy per condition.\n\nBands/bars: descriptive 95% Wilson intervals.\nAdaptive line represents a moving policy,\nnot a fixed position.\n\nScreen selects controls; fresh confirmation\ndetermines non-inferiority.',va='top',transform=axes.flat[-1].transAxes)
 fig.suptitle('Screen: does the fixed-position frontier depend on trading conditions?',fontsize=15)
 fig.tight_layout();fig.savefig(path/'screen-curves.png',dpi=160);fig.savefig(path/'screen-curves.svg');plt.close(fig)
 data=confirmation or screen
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
 bins=['0-19','20-39','40-59','60-79','80-99']
 for ax,regimes in zip(axes,[('none','moderate','heavy','full'),('mixed','rising','falling')]):
  for regime in regimes:
   d=next(r for r in data['diagnostics'] if r['regime']==regime)
   y=[d['bins'][b]['mean_alpha'] for b in bins]
   ax.plot([10,30,50,70,90],y,'o-',label=regime)
  ax.set_xlabel('Global turn (20-turn bins)');ax.set_ylim(0,1);ax.grid(alpha=.15);ax.legend(frameon=False)
  ax.axvline(40,color='gray',linestyle=':',alpha=.7)
 axes[0].set_ylabel('Mean applied slider position')
 fig.suptitle(f"Adaptive response ({data['stage']}): means among games reaching each bin")
 fig.tight_layout();fig.savefig(path/'activity-trajectories.png',dpi=160);fig.savefig(path/'activity-trajectories.svg');plt.close(fig)
 if confirmation:
  fig,ax=plt.subplots(figsize=(10,7))
  rows=[r for r in confirmation['per_regime'] if r['comparator']=='conditional']
  labels=[f"{r['regime']} (fixed {r['control']})" for r in rows]
  for name,label in [('global','Aggregate: fixed .5'),('conditional','Aggregate: condition controls')]:
   rows.append(confirmation['primary'][name]);labels.append(label)
  for y,r in enumerate(rows):
   d=100*r['difference'];lo,hi=[100*v for v in r['interval95']]
   ax.errorbar(d,y,xerr=[[d-lo],[hi-d]],fmt='o',color='#245985' if y<7 else '#b95826',capsize=4)
  ax.set_yticks(range(len(rows)),labels);ax.invert_yaxis()
  ax.axvline(0,color='gray',linewidth=1);ax.axvline(-2,color='#b95826',linestyle='--',label='−2 pp margin')
  ax.set_xlabel('Adaptive minus fixed win rate (percentage points)')
  ax.set_title('Fresh adaptive-minus-fixed comparisons');ax.legend();ax.grid(axis='x',alpha=.15)
  fig.text(.5,.01,'Condition intervals are descriptive; aggregate lower bounds are the registered tests.',ha='center',fontsize=9)
  fig.tight_layout(rect=(0,.035,1,1));fig.savefig(path/'confirmation-contrasts.png',dpi=160);fig.savefig(path/'confirmation-contrasts.svg');plt.close(fig)



if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--screen',type=Path,required=True);p.add_argument('--confirmation',type=Path);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 plot(json.loads(a.screen.read_text()),a.out,json.loads(a.confirmation.read_text()) if a.confirmation else None)
