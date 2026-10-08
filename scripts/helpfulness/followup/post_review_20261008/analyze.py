import sys,json,hashlib
from pathlib import Path
from collections import Counter
from html.parser import HTMLParser
R=Path(__file__).resolve().parent; B=R.parent; W=B.parent
sys.path.insert(0,str(W/'upload_preparation/python_deps')); sys.path.insert(0,str(B))
import duckdb
from prepare_helpfulness import readable
OUT=R/'results'; OUT.mkdir(exist_ok=True)
def read(p): return json.loads(p.read_text(encoding='utf8'))
def save(n,v): (OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
freeze=read(B/'gpt_case_review_20261008/freeze_receipt.json')
for n,h in freeze['hashes'].items(): assert hashlib.sha256((B/'gpt_case_review_20261008'/n).read_bytes()).hexdigest()==h
cases=read(B/'gpt_case_review_20261008/case_results.json')
def labels(c,model):
    if model=='GPT': return [a['gpt']['label'] if a['gpt']['label']!='uncertain' else None for a in c['answers']]
    return [a['qwen'][model]['label'] if a['qwen'][model]['status']=='valid' and a['qwen'][model]['label']!='uncertain' else None for a in c['answers']]
def stats(cc,model,pos):
    complete=[]; patterns=Counter(); bounds=[0,0]
    for c in cc:
        ls=labels(c,model); vs=[None if x is None else x in pos for x in ls]
        possibilities=[(a,b) for a in ([False,True] if vs[0] is None else [vs[0]]) for b in ([False,True] if vs[1] is None else [vs[1]])]
        transitions=[int(not a and b) for a,b in possibilities]
        bounds[0]+=min(transitions); bounds[1]+=max(transitions)
        if None in vs: continue
        a,b=vs; complete.append((a,b)); patterns[str(int(a))+' -> '+str(int(b))]+=1
    return {'candidate_n':len(cc),'complete_n':len(complete),'missing_n':len(cc)-len(complete), 'first_positive':sum(a for a,b in complete),'either_positive':sum(a or b for a,b in complete),'additional_positive':sum(not a and b for a,b in complete),'patterns':dict(patterns),'candidate_additional_positive_bounds':bounds}
context_flags={'edited_question_scope','missing_linked_figures','incomplete_input_layout','missing_screenshot','possible_code_formatting_loss','linked_content_not_read'}
subsets={'all31':cases,'added14':[c for c in cases if c['added']],'controls8':[c for c in cases if c['control']],
    'common_valid_all_models':[c for c in cases if all(None not in labels(c,m) for m in ['GPT','14B','7B'])],
    'no_recorded_context_flag':[c for c in cases if not(set(c['flags'])&context_flags)],
    'GPT_both_high_confidence':[c for c in cases if all(a['gpt']['confidence']=='high' for a in c['answers'])]}
cells=[]
for name,cc in subsets.items():
    for threshold,pos in [('substantial',{'substantial'}),('any_help',{'partial','substantial'})]:
        for model in ['GPT','14B','7B']: cells.append(dict(subset=name,model=model,threshold=threshold,**stats(cc,model,pos)))
save('threshold_sensitivity.json',cells)
class Assets(HTMLParser):
    def __init__(self): super().__init__(convert_charrefs=True); self.links=[];self.images=[];self.tables=0;self.pre=[];self.inpre=False
    def handle_starttag(self,t,a):
        d=dict(a)
        if t=='a' and d.get('href'):self.links.append(d['href'])
        if t=='img':self.images.append({k:d.get(k,'') for k in ['src','alt','title']})
        if t=='table':self.tables+=1
        if t=='pre':self.inpre=True;self.pre.append('')
        if self.inpre and t=='br':self.pre[-1]+='\n'
    def handle_endtag(self,t):
        if t=='pre':self.inpre=False
    def handle_data(self,d):
        if self.inpre:self.pre[-1]+=d
tasks=read(B/'answer_trajectory_20261007/tasks.json'); meta=read(B/'answer_trajectory_20261007/answer_metadata.json')
qtexts={}; atexts={}
for t in tasks:
    d=json.loads(t['messages'][1]['content'].split('DATA:\n',1)[1]); q=t['question_id']
    assert q not in qtexts or qtexts[q]==d['question']
    qtexts[q]=d['question']; atexts[t['pair_id']]=t['lines']
byanswer={m['answer_id']:m for m in meta}; reviewedq={c['question_id']:c for c in cases}; reviewedanswers={a['metadata']['answer_id'] for c in cases for a in c['answers']}
con=duckdb.connect();con.execute("SET threads=2");con.execute("SET memory_limit='2GB'")
audits=[]; snapshots=[]
for kind,ids,key,file in [('question',qtexts,'question_id','questions_clean.parquet'),('answer',byanswer,'answer_id','answers_clean.parquet')]:
    con.read_parquet(str(W/'sosc314-stackoverflow/data/processed'/file)).create_view('source_'+kind)
    con.execute('CREATE TEMP TABLE selected_'+kind+'(id BIGINT)');con.executemany('INSERT INTO selected_'+kind+' VALUES (?)',[(i,) for i in ids])
    rr=con.execute('SELECT s.'+key+',s.body_html FROM source_'+kind+' s JOIN selected_'+kind+' k ON s.'+key+'=k.id').fetchall()
    assert len(rr)==len(ids)
    for ident,html in rr:
        text=readable(html);p=Assets();p.feed(html or '')
        if kind=='question': assert text==qtexts[ident]; q=ident; keep=q in reviewedq
        else:
            m=byanswer[ident];q=m['question_id'];keep=ident in reviewedanswers
            assert {f'A{i+1}':l for i,l in enumerate(text.splitlines()) if l.strip()}==atexts[m['pair_id']]
        missinglinks=[s for s in p.links if s not in text]
        missingimages=[im for im in p.images if im['src'] and im['src'] not in text]
        lostalt=[im['alt'] for im in p.images if im['alt'] and im['alt'] not in text]
        audit=dict(kind=kind,id=ident,question_id=q,html_sha256=hashlib.sha256((html or '').encode()).hexdigest(),readable_sha256=hashlib.sha256(text.encode()).hexdigest(),href_n=len(p.links),href_not_visible_n=len(missinglinks),image_n=len(p.images),image_source_not_visible_n=len(missingimages),alt_not_visible_n=len(lostalt),table_n=p.tables,pre_blocks=len(p.pre),pre_newline_counts=[s.count('\n') for s in p.pre],input_reproduced=True)
        audits.append(audit)
        if keep:snapshots.append(dict(**audit,case_id=reviewedq[q]['case_id'],body_html=html,readable=text,missing_href=missinglinks,images=p.images,missing_alt=lostalt))
save('input_audit.json',audits);save('reviewed_source_snapshots.json',snapshots)
summary={}
for kind in ['question','answer']:
    rr=[a for a in audits if a['kind']==kind]
    summary[kind]={'n':len(rr),'reproduced_n':sum(a['input_reproduced'] for a in rr)}
    for k in ['href_not_visible_n','image_source_not_visible_n','alt_not_visible_n','table_n']:summary[kind][k]={'documents':sum(a[k]>0 for a in rr),'instances':sum(a[k] for a in rr)}
save('summary.json',{'threshold_cells':len(cells),'input_audit':summary,'reviewed_question_ids':list(reviewedq),'context_flag_exclusions':[c['case_id'] for c in cases if set(c['flags'])&context_flags], 'limits':['Missing HTML attributes are not automatically material semantic losses.','Archived HTML may itself already lack original content.','No external linked content retrieved; no inference of image contents.','Post-hoc exclusions are descriptive, not corrected estimates.','Threshold changes define different outcomes, not improved accuracy.','No GPT labels extrapolated to full cohort; no model labels overwritten.'],'completed':True})
save('manifest.json',{'plan_sha256':hashlib.sha256((R/'plan.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'review_receipt_sha256':hashlib.sha256((B/'gpt_case_review_20261008/freeze_receipt.json').read_bytes()).hexdigest(),'source_parquet_metadata':{f:{'size':(W/'sosc314-stackoverflow/data/processed'/f).stat().st_size} for f in ['questions_clean.parquet','answers_clean.parquet']},'reserved_used':False})
print(json.dumps(read(OUT/'summary.json'),indent=2))
print(json.dumps([c for c in cells if c['subset'] in ['all31','common_valid_all_models']],indent=2))
