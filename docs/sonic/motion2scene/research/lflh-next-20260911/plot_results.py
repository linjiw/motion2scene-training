"""Export descriptive scientific figure from the retained architecture CSV."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT/'benchmark/architecture-results.csv').open()))
names = ['unconditional','mlp','conv','transformer']
fig, axes = plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
for ax, metric, title in zip(axes,
        ['capsule_clear_count','reconstruction_count','capsule_contrast_count'],
        ['Capsule clearance at recorded frames','Target reconstruction','Capsule contrast witness']):
    for i,name in enumerate(names):
        fits = [r for r in rows if r['architecture']==name]
        values = [100*int(r[metric])/int(r['draws']) for r in fits]
        ax.bar(i,np.mean(values),color='#ced4da',width=.65)
        ax.scatter([i-.08,i+.08],values,color='#1b4965',s=25,zorder=3)
    ax.set_xticks(range(4),['Uncond.','MLP','CNN','Transformer'],rotation=25,ha='right')
    ax.set_ylim(0,105); ax.set_title(title,fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylabel('Percent of 56 draws per fit')
fig.suptitle('In-bank geometry diagnostic — no physical passage or transfer measurement',fontsize=12)
fig.savefig(ROOT/'architecture-diagnostic.png',dpi=180)
fig.savefig(ROOT/'architecture-diagnostic.pdf')
