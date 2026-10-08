"""Deterministic static SVG figures from versioned JSON; no generative images.
Standard-library only. PNG copies may be rendered from these SVGs with sharp.
"""
import json,gzip,html
from pathlib import Path
R=Path(__file__).resolve().parents[2]
D=R/'tables/annotation_diagnostics/followup'; F=R/'figures/annotation_diagnostics/followup'
F.mkdir(parents=True,exist_ok=True)
def read(rel):
    p=D/rel
    if p.exists():return json.loads(p.read_text(encoding='utf8'))
    return json.loads(gzip.decompress(p.with_name(p.name+'.gz').read_bytes()))
def begin(w,h):return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/><g font-family="Arial, sans-serif" fill="#182733">']
def text(s,x,y,t,size=14,anchor='start',fill='#182733'):s.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}">{html.escape(str(t))}</text>')
def line(s,x1,y1,x2,y2,color='#cad2d9',width=1):s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')
def finish(s,name):s.append('</g></svg>');(F/name).write_text('\n'.join(s),encoding='utf8')
data=read('association_sensitivity_20261007/results/analysis.json');cells=[c for c in data['cells'] if c['phase']=='pooled']
assert len(cells)==7
s=begin(1100,800);text(s,30,34,'Intent–first-answer associations depend on measurement choices',23)
text(s,30,61,'Pooled reused development sample; substantial-rated first answers; unadjusted descriptive differences',14)
colors=['#25618b','#d28228','#268776','#9d4a91']
names=['Direct intent / original examples','Direct intent / no examples','Request-first intent / original examples','Request-first intent / no examples']
for i,(color,name) in enumerate(zip(colors,names)):
    x=30+(i%2)*520;y=94+(i//2)*28;line(s,x,y-5,x+20,y-5,color,4);text(s,x+30,y,name,14)
x0=265;w=760
def xpos(v):return x0+(v+1)*w/2
for tick in [-1,-.5,0,.5,1]:
    x=xpos(tick);line(s,x,154,x,675,'#80929e' if tick==0 else '#e4e9ed');text(s,x,701,str(int(tick*100)),13,'middle')
for i,c in enumerate(cells):
    y=178+i*76;text(s,28,y+14,c['category'].replace('_',' ').title(),16);text(s,28,y+35,f"Matched eligible n = {c['eligible_n']}",12)
    for j,p in enumerate(c['pipelines']):
        yy=y+j*13;v=p['risk_difference'];ci=p['conditional_bootstrap95']
        if v is None:text(s,290,yy+4,f'{j+1}: suppressed (intent group < 10)',11,fill=colors[j]);continue
        assert ci is not None and -1<=v<=1
        line(s,xpos(ci[0]),yy,xpos(ci[1]),yy,colors[j],2)
        s.append(f'<circle cx="{xpos(v)}" cy="{yy}" r="4" fill="{colors[j]}"/>')
text(s,645,729,'Difference: intent present minus absent (percentage points)',14,'middle')
text(s,30,761,'Intervals: paired bootstrap conditional on fixed model labels, not annotation uncertainty or population inference.',12)
text(s,30,781,'Intent categories overlap. No causal interpretation; these are sensitivity results, not validated effects.',12)
finish(s,'intent_association_sensitivity.svg')
cases=read('gpt_case_review_20261008/case_results.json');assert len(cases)==31
s=begin(1050,1130);text(s,28,34,'Case-level helpfulness judgments do not consistently agree',23)
text(s,28,62,'31 targeted/control questions; two answers each in chronological order; GPT is not human ground truth',14)
palette={'substantial':'#247b77','partial':'#e5ac43','unhelpful':'#b85151','uncertain':'#8978a7','invalid':'#cbd2d8'}
for i,(label,color) in enumerate(palette.items()):
    x=28+i*200;s.append(f'<rect x="{x}" y="80" width="16" height="16" fill="{color}"/>');text(s,x+24,94,label,13)
models=['14B','7B','GPT'];columns=[270,490,710]
for x,m in zip(columns,models):
    text(s,x+65,132,m,17,'middle');text(s,x+28,154,'First',12,'middle');text(s,x+101,154,'Second',12,'middle')
text(s,28,151,'Case / selection',14);text(s,915,151,'Context flag*',12,'middle')
flags={'edited_question_scope','missing_linked_figures','incomplete_input_layout','missing_screenshot','possible_code_formatting_loss','linked_content_not_read'}
for i,c in enumerate(cases):
    y=169+i*27;text(s,28,y+16,c['case_id']+('  added14' if c['added'] else '  control' if c['control'] else '  disagreement'),13)
    for m,x in zip(models,columns):
        for rank,a in enumerate(c['answers']):
            label=a['gpt']['label'] if m=='GPT' else a['qwen'][m]['label'] if a['qwen'][m]['status']=='valid' else 'invalid'
            s.append(f'<rect x="{x+rank*73}" y="{y}" width="56" height="20" rx="2" fill="{palette[label]}"/>')
    if set(c['flags'])&flags:text(s,915,y+16,'*',17,'middle')
text(s,28,1038,'Selection groups overlap: added14 and model-disagreement14 share five cases; eight controls are non-target cases.',12)
text(s,28,1060,'Grey means invalid/skipped, not unhelpful. One GPT answer was uncertain. Read reasons and evidence before interpreting.',12)
text(s,28,1082,'* Recorded edited/missing-context flags. No source post code was executed. This is not a population accuracy estimate.',12)
text(s,28,1104,'Ratings were frozen before revealing answer order and prior labels; the reviewer already knew aggregate findings.',12)
finish(s,'gpt_case_review_matrix.svg')
print('Generated two data-derived SVG figures')
