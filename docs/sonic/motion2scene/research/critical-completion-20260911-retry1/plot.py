"""Descriptive plots; no model selection or training."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

out = Path(__file__).resolve().parent
rows = json.loads((out/'evaluation-results.json').read_text())['rows']
methods = ['unconditional','cover','masked','contrast','combined']
labels = ['Unconditional','Coverage','+ Masked','+ Contrast','Both']
fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,key,title in zip(axes,['critical_mass','penetration_witness_mass'],
                       ['Critical-location probability','Penetration-witness probability']):
    for i,m in enumerate(methods):
        values=[100*r[key] for r in rows if r['method']==m and r['condition']=='conditioned']
        ax.bar(i,np.mean(values),color='#c8d6df')
        ax.scatter([i-.07,i+.07],values,color='#184e77',s=28,zorder=3)
    ax.set_xticks(range(len(methods)),labels,rotation=25,ha='right')
    ax.set_ylim(0,60);ax.set_ylabel('Probability (%)');ax.set_title(title)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Withheld coordinates, same motion bank — geometry only')
fig.savefig(out/'comparison.png',dpi=170)
fig.savefig(out/'comparison.pdf')

fields=torch.load(out/'fields.pt',weights_only=True)
prob=torch.load(out/'probabilities.pt',weights_only=True)
crit=(fields['lower']>=.01)&(fields['upper']<=-.01).any(0,keepdim=True)
test=fields['test']
fig,axes=plt.subplots(7,3,figsize=(10,15),layout='constrained')
for i,name in enumerate(fields['names']):
    images=[np.where(test.numpy(),crit[i].numpy().astype(float),np.nan)]
    for method in ['cover','contrast']:
        value=(prob[f'{method}-301-conditioned'][i]+prob[f'{method}-302-conditioned'][i])/2
        a=np.full(tuple(test.shape),np.nan);a[test.numpy()]=value.numpy()
        images.append(a)
    for j,a in enumerate(images):
        ax=axes[i,j]
        im=ax.imshow(a.T,origin='lower',extent=[1,4.5,.7,1.65],aspect='auto',
                     cmap=ListedColormap(['#dddddd','#111111']) if j==0 else 'viridis',vmin=0,vmax=1 if j==0 else None)
        if i==0:ax.set_title(['Computed capsule critical cells','Coverage distribution','Contrast distribution'][j])
        if j==0:ax.set_ylabel(name+'\nBeam underside (m)',fontsize=8)
        if i==6:ax.set_xlabel('Beam position x (m)')
        if j>0:fig.colorbar(im,ax=ax,fraction=.04,pad=.02)
fig.suptitle('All seven targets; white blocks were training coordinates\nProbability maps average two fits; each probability color scale is labeled',fontsize=11)
fig.savefig(out/'critical-maps.png',dpi=130)
fig.savefig(out/'critical-maps.pdf')
with (out/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
