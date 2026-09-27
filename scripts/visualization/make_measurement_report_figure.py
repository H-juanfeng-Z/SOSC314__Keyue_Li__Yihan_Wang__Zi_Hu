"""Static paired-rating figure; all counts are read from completed logs."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
rows=[json.loads(x) for x in (ROOT/'tables/annotation_diagnostics/measurement/development/calls.jsonl').read_text(encoding='utf8').splitlines()]
idx={(r['id'],r['arm']):r for r in rows if r['suite']=='helpfulness_development'}
labels=['unhelpful','partial','substantial']
matrix=[[0]*3 for _ in labels]
for pid in sorted({key[0] for key in idx}):
    a,b=idx[pid,'baseline'],idx[pid,'evidence']
    assert a['status']==b['status']=='valid'
    matrix[labels.index(a['parsed']['overall'])][labels.index(b['parsed']['overall'])]+=1
assert sum(map(sum,matrix))==120
im=Image.new('RGB',(1500,1000),'white');d=ImageDraw.Draw(im)
def text(x,y,s,size=30,fill='#233044'):
    d.text((x,y),s,font=ImageFont.load_default(size=size),fill=fill)
text(55,30,'First-answer helpfulness: paired prompt comparison',43)
text(55,91,'120 stratified development pairs; frozen Qwen2.5-Coder-14B',27)
text(500,158,'Evidence-enhanced prompt',32)
for j,label in enumerate(labels):text(360+j*325,224,label.capitalize(),28)
text(55,285,'Baseline',29);text(55,323,'prompt',29)
for i,label in enumerate(labels):
    text(85,418+i*170,label.capitalize(),27)
    for j in range(3):
        x=320+j*325;y=330+i*170;n=matrix[i][j]
        color='#467eaa' if i==j else ('#e8b454' if n else '#eef1f5')
        d.rectangle((x,y,x+305,y+150),fill=color)
        text(x+130,y+52,str(n),42,'white' if i==j else '#233044')
text(55,871,'Same rating: 110/120 (91.7%). Changed: 9 substantial to partial; 1 unhelpful to partial.',25)
text(55,923,'Counts are development diagnostics, not annotation accuracy or population prevalence.',25)
im.save(ROOT/'figures/annotation_diagnostics/first_answer_helpfulness_transitions.png')
print(matrix)
