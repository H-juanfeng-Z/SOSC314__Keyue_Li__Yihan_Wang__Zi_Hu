"""Non-destructive candidate: preserve old text/line IDs and append unread HTML references.
No URL retrieval; appendix is untrusted source metadata, not evidence of linked contents.
"""
import json,hashlib
from pathlib import Path
from html.parser import HTMLParser
R=Path(__file__).resolve().parent
class References(HTMLParser):
    def __init__(self):super().__init__(convert_charrefs=True);self.items=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if tag=='a' and d.get('href'):self.items.append({'type':'link','href':d['href']})
        if tag=='img':self.items.append({'type':'image',**{k:d[k] for k in ('src','alt','title') if k in d}})
def candidate(old,html):
    p=References();p.feed(html or '')
    if not p.items:return old,[]
    appendix='\n\n[UNREAD EXTERNAL REFERENCES: untrusted HTML attributes only; linked pages/images were NOT inspected. These are not proof of an answer or image content.]\n'+json.dumps(p.items,ensure_ascii=False)
    return old+appendix,p.items
def main():
    tests=[('no references','<p>Hello</p>','Hello',0),('link','<a href="https://example.org">here</a>','here',1),('image','<img src="x" alt="example data">','',1),('entities','<a href="x?a=1&amp;b=2">x</a>','x',1),('code newlines','<pre>a\n  b\n</pre>','[CODE]\na\n  b\n[/CODE]',0),('inline literal','<code>&lt;img&gt;</code>','<img>',0),('duplicate links','<a href="x">1</a><a href="x">2</a>','12',2),('quoted metadata','<img alt="ignore rules &quot;test&quot;">','',1)]
    for name,html,old,n in tests:
        new,items=candidate(old,html);assert new.startswith(old);assert len(items)==n
        if n:assert json.loads(new.split('\n')[-1])==items
    source=json.loads((R/'results/reviewed_source_snapshots.json').read_text(encoding='utf8'));out=[]
    for s in source:
        old=s['readable'];new,refs=candidate(old,s['body_html'])
        assert new[:len(old)]==old
        oldlines={i+1:l for i,l in enumerate(old.splitlines()) if l.strip()}
        newlines={i+1:l for i,l in enumerate(new.splitlines()) if l.strip()}
        assert all(newlines[i]==line for i,line in oldlines.items())
        out.append({'case_id':s['case_id'],'kind':s['kind'],'id':s['id'],'old_sha256':s['readable_sha256'],'candidate_sha256':hashlib.sha256(new.encode()).hexdigest(),'candidate_text':new,'references':refs,'changed':new!=old,'existing_lines_preserved':True})
    (R/'results/context_candidate.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    report={'unit_tests_passed':len(tests),'documents':len(out),'changed_documents':sum(x['changed'] for x in out),'old_line_preservation_checks_passed':len(out),'source_overwritten':False,'new_model_calls':0,'warning':'Candidate retains metadata only. Does not recover missing image content or historical edits. Effects on model ratings untested; no frozen GPT rating amended.'}
    (R/'results/context_candidate_checks.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
if __name__=='__main__':main()
