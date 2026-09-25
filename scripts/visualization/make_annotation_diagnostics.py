"""Static diagnostic plots from versioned JSON summaries (Pillow + SVG)."""
import html
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'figures/annotation_diagnostics'
OUT.mkdir(parents=True,exist_ok=True)

class Canvas:
    def __init__(self,height):
        self.im=Image.new('RGB',(1200,height),'white')
        self.draw=ImageDraw.Draw(self.im)
        self.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}"><rect width="100%" height="100%" fill="white"/>']
    def text(self,x,y,s,size=20,color='#233044'):
        self.draw.text((x,y),s,font=ImageFont.load_default(size=size),fill=color)
        self.svg.append(f'<text x="{x}" y="{y+size}" font-family="sans-serif" font-size="{size}" fill="{color}">{html.escape(s)}</text>')
    def rect(self,x,y,w,h,color):
        self.draw.rectangle((x,y,x+w,y+h),fill=color)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>')
    def save(self,name):
        self.im.save(OUT/(name+'.png'))
        (OUT/(name+'.svg')).write_text('\n'.join(self.svg+['</svg>']),encoding='utf8')

def main():
    data=json.loads((ROOT/'tables/annotation_diagnostics/intent_stability/results.json').read_text(encoding='utf8'))
    c=Canvas(660)
    c.text(40,20,'Question-intent labels: sensitivity to prompts and input',28)
    c.text(40,63,'Exact seven-label set agreement on jointly valid binary real questions',19)
    comparisons=[('Prompt package: shared vs isolated','full-shared_json-greedy__full-isolated_json-greedy'),
                 ('Input: full vs prose (shared)','full-shared_json-greedy__prose-shared_json-greedy'),
                 ('Input: full vs title (isolated)','full-isolated_json-greedy__title-isolated_json-greedy'),
                 ('Decoding: greedy vs sample314','full-isolated_json-greedy__full-isolated_json-sample314')]
    for i,(label,key) in enumerate(comparisons):
        y=110+i*110
        c.text(40,y,label,20)
        for j,(model,color) in enumerate([('7b','#437ab5'),('14b','#dd8845')]):
            r=data['models'][model]['paired_comparisons'][key]
            yy=y+30+j*29
            c.text(40,yy,model.upper(),17,color)
            value=r['exact_label_set_agreement']
            c.rect(110,yy+3,650,20,'#edf0f3')
            c.rect(110,yy+3,650*value,20,color)
            c.text(785,yy,f'{value:.1%}   n={r["joint_valid_binary_questions"]}; excluded={r["excluded_invalid_uncertain_or_missing"]}',17)
    c.text(110,555,'0%',16); c.text(723,555,'100%',16)
    c.text(40,591,'Development diagnostics only: agreement is not accuracy. Denominators differ by comparison.',17)
    c.text(40,619,'Qwen2.5-Coder; full means processed text, not raw HTML. No held-out cases used.',17)
    c.save('intent_stability')

    data=json.loads((ROOT/'tables/annotation_diagnostics/helpfulness_summary.json').read_text(encoding='utf8'))['rubric_factorial']['cells']
    c=Canvas(900)
    c.text(40,20,'First-answer helpfulness: fixed-model prompt diagnostics',28)
    c.text(40,64,'Qwen2.5-Coder-14B | 13 code-containing development pairs | 208 schema-valid calls',18)
    colors={'unhelpful':'#597797','partial':'#edbd59','substantial':'#5b9b83','uncertain':'#a6a6a6'}
    for j,(label,color) in enumerate(colors.items()):
        c.rect(40+j*270,105,18,18,color);c.text(67+j*270,101,label,18)
    for i,arm in enumerate(['baseline','evidence','dimensions','combined']):
        y=155+i*162
        c.text(40,y,arm.capitalize(),22)
        for j,condition in enumerate(['original','swapped_code','code_only','foreign_code_only']):
            row=data[arm+'/'+condition]; yy=y+31+j*29
            c.text(40,yy,condition.replace('_',' '),17)
            x=250
            for label,color in colors.items():
                n=row['overall'].get(label,0);w=55*n
                if n:
                    c.rect(x,yy,w,24,color)
                    c.text(x+w/2-7,yy+1,str(n),16,'#ffffff' if label!='partial' else '#233044')
                x+=w
            c.text(1000,yy,f'n={row["valid"]}',17)
    c.text(40,822,'Bars show counts, not correctness. Correct prose may remain useful after code replacement.',17)
    c.text(40,852,'Adaptive development tests; no independent semantic gold labels or held-out validation.',17)
    c.save('helpfulness_prompt_sensitivity')

if __name__=='__main__':main()
