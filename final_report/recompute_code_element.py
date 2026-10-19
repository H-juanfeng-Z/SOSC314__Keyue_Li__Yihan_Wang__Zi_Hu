"""Reproduce code-element bin statistics without model inference or score changes."""
from pathlib import Path
import argparse, hashlib, json, math, statistics

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def bin_name(n):
    assert isinstance(n,int) and n>=0
    return '0' if n==0 else '1-2' if n<=2 else '3-5' if n<=5 else '6-10' if n<=10 else '11+'

def summarize(rows):
    assert len(rows)==len({r['question_id'] for r in rows})==1000
    result={'selected_questions':len(rows),'variable':'code_element_count: number of <code> elements, including inline and block code; not code length',
            'bins':['0','1-2','3-5','6-10','11+'],'error_bars':'mean +/- 1 standard error; sample SD (ddof=1) / sqrt(n); not 95% confidence intervals',
            'support':'Per-arm valid numeric scores from the fixed 1000-question fine sample; not restricted to identical three-score-valid support.','arms':{}}
    for arm in ['format','scope']:
        groups=[]
        for label in result['bins']:
            values=[r[arm] for r in rows if r[arm] is not None and bin_name(r['code_element_count'])==label]
            assert values and all(type(x) is int and 0<=x<=10 for x in values)
            n=len(values);mean=statistics.mean(values);sd=statistics.stdev(values);se=sd/math.sqrt(n)
            groups.append({'bin':label,'n':n,'mean':mean,'sample_sd':sd,'se':se,'mean_minus_se':mean-se,'mean_plus_se':mean+se})
        result['arms'][arm]={'valid_n':sum(x['n'] for x in groups),'not_valid_n':len(rows)-sum(x['n'] for x in groups),'groups':groups}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--support',type=Path,default=Path(__file__).with_name('code_element_support.json'))
    p.add_argument('--fine',type=Path);p.add_argument('--questions',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--verify',type=Path)
    a=p.parse_args()
    if a.fine or a.questions:
        assert a.fine and a.questions,'Provide both --fine and --questions'
        import pyarrow.parquet as pq
        fine=json.loads(a.fine.read_text(encoding='utf8'));assert len(fine)==1000
        metadata=pq.read_table(a.questions,columns=['question_id','code_element_count'],filters=[('question_id','in',[r['question_id'] for r in fine])]).to_pylist()
        byid={r['question_id']:r['code_element_count'] for r in metadata};assert len(byid)==len(metadata)==1000
        rows=[]
        for r in fine:
            out={'question_id':r['question_id'],'answer_id':r['answer_id'],'code_element_count':byid[r['question_id']]}
            for arm in ['format','scope']:
                s=r[arm];out[arm]=s['score'] if s['status']=='valid' and s['usable'] and s['score'] is not None else None
            rows.append(out)
    else:rows=json.loads(a.support.read_text(encoding='utf8'))
    result=summarize(rows)
    if a.verify:
        expected=json.loads(a.verify.read_text(encoding='utf8'))
        assert result==expected,'Bin statistics differ from the released summary'
        print('All 10 bin means, sample sizes, SDs and SEs match the released summary.')
    else:print(json.dumps(result,ensure_ascii=False,indent=2))
    if a.output:a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
