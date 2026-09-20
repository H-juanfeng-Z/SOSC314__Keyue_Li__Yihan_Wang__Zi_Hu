"""Rebuild the published figures from their aggregate numerical results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'report_outputs'
data = json.loads((OUT / 'figure_data.json').read_text(encoding='utf8'))
metrics = data['metrics']
categories = ['api_usage','discrepancy','errors','review','conceptual','api_change','learning']
names = ['Implementation','Unexpected behavior','Explicit error','Review / comparison','Concept explanation','API / version change','Learning resources']
variant_names = {'examples':'Baseline','shared_json':'+ JSON examples','isolated':'Single-target','isolated_json':'Single-target + JSON'}
labels = [m['model'] + '  ' + variant_names[m['variant']] for m in metrics]
heat = [[m['valid_yes_counts'][c] for c in categories] for m in metrics]
missing = [m['real_invalid'] for m in metrics]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
y=np.arange(8)
fig,axs=plt.subplots(1,2,figsize=(13,6),gridspec_kw={'width_ratios':[1,1.2]})
ax=axs[0];vals=[m['valid']/1036*100 for m in metrics]
ax.scatter(vals,y,s=55,color='#256b91');ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,115)
for i,m in enumerate(metrics):ax.text(vals[i]+2,i,f"{m['valid']}/1036",va='center',fontsize=9)
ax.set_title('A  Output validation',loc='left',fontweight='bold');ax.set_xlabel('Strict schema and evidence-ID pass rate (%)')
ax=axs[1]
for key,offset,color,marker,label in [('decision_constraints',-.12,'#b77622','s','Decision only'),('strict_constraints',.12,'#256b91','o','Decision + output validation')]:
    vals=[m[key]/52*100 for m in metrics];ax.scatter(vals,y+offset,s=45,c=color,marker=marker,label=label)
    for i,m in enumerate(metrics):ax.text(vals[i]+1.5,i+offset,f"{m[key]}/52",va='center',fontsize=8)
ax.set_yticks(y,[]);ax.invert_yaxis();ax.set_xlim(0,118);ax.set_xlabel('Synthetic constraint compliance (%)');ax.set_title('B  Synthetic diagnostic checks',loc='left',fontweight='bold')
ax.legend(loc='lower left',bbox_to_anchor=(0,-.29),frameon=False,fontsize=9)
for ax in axs:ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x',alpha=.18);ax.axhline(3.5,color='#aaaaaa',lw=.7)
fig.suptitle('Question-intent annotation: output reliability and diagnostic behavior',fontsize=14,fontweight='bold')
fig.text(.02,.015,'148 questions per arm: 120 real + 28 synthetic; 7 category decisions per question.\n52 checks = 28 intended-positive + 24 learning-resource-negative constraints. AI-authored, shared templates; NOT real-world accuracy.',fontsize=9)
fig.subplots_adjust(left=.24,right=.98,top=.87,bottom=.24,wspace=.15)
for ext in ['png','svg']:fig.savefig(OUT/f'figure1_intent_validation.{ext}',dpi=200)
plt.close(fig)
fig,ax=plt.subplots(figsize=(12,6.3));im=ax.imshow(heat,cmap='Blues',vmin=0,vmax=120,aspect='auto')
ax.set_yticks(y,labels);ax.set_xticks(np.arange(7),names,rotation=28,ha='right')
for i in range(8):
    for j in range(7):ax.text(j,i,str(heat[i][j]),ha='center',va='center',color='white' if heat[i][j]>65 else '#16334b')
ax.set_title('Prompt sensitivity on 120 real development questions',loc='left',fontweight='bold',pad=24)
for i,n in enumerate(missing):ax.text(7.02,i,f'{n}/840',va='center',fontsize=9,clip_on=False)
ax.text(7.02,-.6,'Invalid calls',fontsize=9,clip_on=False)
fig.text(.02,.025,'Each cell: valid yes count out of 120 questions (darker = more). Not prevalence or accuracy estimates; invalid outputs are not no.\nDifferent prompt packages can change label frequency; single-target also changes wording. Multi-label classification is allowed.',fontsize=9)
fig.subplots_adjust(left=.25,right=.84,top=.9,bottom=.3)
for ext in ['png','svg']:fig.savefig(OUT/f'figure2_intent_label_sensitivity.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
print(json.dumps(metrics,indent=2))
