"""Prepare development-only inputs. No reserved cases sent to a model."""
import collections
import hashlib
import json
import sys
from pathlib import Path

import argparse
p=argparse.ArgumentParser()
p.add_argument('--workspace',type=Path,required=True,help='Original data workspace layout; see documentation')
a=p.parse_args()
WORK=a.workspace.resolve()
import duckdb
from prepare_helpfulness import readable

OUT=WORK/'sosc314_week4_experiments/measurement_20260926/data'
SOURCE=WORK/'sosc314-stackoverflow/data/processed'
OLD=WORK/'sosc314_week4_experiments/data/blinded_pairs.json'
CASES=WORK/'sosc314_week4_experiments/measurement_20260926/source_cases.json'

def dump(name,data):
    with (OUT/name).open('x',encoding='utf8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,default=str)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prior=json.loads(OLD.read_text(encoding='utf8'))
    cases=json.loads(CASES.read_text(encoding='utf8'))
    excluded={int(r['question_id']) for r in prior}
    excluded.update(int(r['question_id']) for r in cases if r.get('question_id') is not None)
    con=duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='2GB'")
    con.read_parquet(str(SOURCE/'questions_clean.parquet')).create_view('q')
    con.read_parquet(str(SOURCE/'answers_clean.parquet')).create_view('a')
    con.execute('CREATE TEMP TABLE excluded(question_id BIGINT)')
    con.executemany('INSERT INTO excluded VALUES (?)',[(i,) for i in sorted(excluded)])
    sql='''WITH first_answer AS (
      SELECT * EXCLUDE(rn) FROM (SELECT *, row_number() OVER
        (PARTITION BY question_id ORDER BY answer_created_at,answer_id) rn FROM a) WHERE rn=1
    ), eligible AS (
      SELECT q.question_id,a.answer_id,q.title,q.body_html question_html,a.body_html answer_html,
        q.creation_year,a.answer_score,CAST(q.question_created_at AS VARCHAR) question_created_at,
        CAST(a.answer_created_at AS VARCHAR) answer_created_at,
        CASE WHEN q.creation_year<=2011 THEN '2008-2011' WHEN q.creation_year<=2014 THEN '2012-2014' ELSE '2015-2016' END year_band,
        CASE WHEN regexp_matches(lower(a.body_html),'<pre(?:\\s|>)') THEN 'code' ELSE 'no_code' END code_band,
        CASE WHEN a.answer_score<0 THEN 'negative' WHEN a.answer_score=0 THEN 'zero'
             WHEN a.answer_score<=5 THEN '1-5' WHEN a.answer_score>5 THEN '6+' ELSE 'missing' END score_band
      FROM q JOIN first_answer a USING(question_id)
      WHERE q.creation_year BETWEEN 2008 AND 2016
        AND q.question_id NOT IN (SELECT question_id FROM excluded)
        AND a.answer_created_at>=q.question_created_at
        AND length(trim(coalesce(q.body_html,'')))>0 AND length(trim(coalesce(a.body_html,'')))>0
    ), ranked AS (
      SELECT *,count(*) OVER(PARTITION BY year_band,code_band,score_band) population_cell,
        row_number() OVER(PARTITION BY year_band,code_band,score_band
          ORDER BY md5(CAST(question_id AS VARCHAR)||':measurement:20260926:314'),question_id) cell_rank
      FROM eligible
    ) SELECT * FROM ranked ORDER BY cell_rank,year_band,code_band,score_band LIMIT 120'''
    cursor=con.execute(sql)
    columns=[c[0] for c in cursor.description]
    selected=[dict(zip(columns,row)) for row in cursor.fetchall()]
    assert len(selected)==120 and len({r['question_id'] for r in selected})==120
    assert not ({r['question_id'] for r in selected}&excluded)
    blinded=[];meta=[]
    for i,r in enumerate(selected):
        pid=f'D26-{i+1:03d}'
        blinded.append({'pair_id':pid,'question_id':r['question_id'],'answer_id':r['answer_id'],
                        'title':r['title'],'question_text':readable(r['question_html']),
                        'answer_text':readable(r['answer_html'])})
        meta.append({'pair_id':pid,**{k:v for k,v in r.items() if k not in ['title','question_html','answer_html']}})
    dump('helpfulness_development120.json',blinded)
    dump('sampling_metadata_private.json',meta)
    # Select BOTH disagreements and agreements, by category, without showing prior labels.
    repo=WORK/'SOSC314__Keyue_Li__Yihan_Wang__Zi_Hu'
    records={}
    for line in (repo/'tables/annotation_diagnostics/decisions/results_ssd/14b/intent/calls.jsonl').open(encoding='utf8'):
        r=json.loads(line)
        if r['source']!='synthetic_diagnostic' and r['arm'] in ['full-shared_json-greedy','full-isolated_json-greedy']:
            records.setdefault((r['case_id'],r['category']),{})[r['arm']]=r
    case_map={r['case_id']:r for r in cases if r['source']!='synthetic_diagnostic'}
    buckets=collections.defaultdict(list)
    for key,arms in records.items():
        if len(arms)!=2 or any(r['status']!='valid' or r['parsed']['decision'] not in ['yes','no'] for r in arms.values()):continue
        values=[r['parsed']['decision'] for r in arms.values()]
        kind='disagreement' if len(set(values))>1 else 'agreement_'+values[0]
        buckets[key[1],kind].append(key)
    tasks=[];selection=[]
    for (cat,kind),keys in sorted(buckets.items()):
        limit=6 if kind=='disagreement' else 2
        chosen=sorted(keys,key=lambda k:hashlib.sha256(('|'.join(k)+':314').encode()).hexdigest())[:limit]
        for key in chosen:
            c=case_map[key[0]]
            tasks.append({'audit_id':f'{key[0]}__{cat}','case_id':key[0],'category':cat,'title':c['title'],'text':c['text']})
            selection.append({'audit_id':tasks[-1]['audit_id'],'selection_group':kind,'prior':records[key]})
    dump('intent_blinded_audit.json',tasks)
    dump('intent_selection_private.json',selection)
    manifest={'purpose':'New development cases and blinded same-model intent audit; not held-out validation',
              'reserved_used':False,'new_pairs':120,'excluded_question_count':len(excluded),
              'sampling':'Round-robin deterministic hash-ranked year x answer-code x score strata; not population representative',
              'score_hidden_from_model':True,'intent_tasks':len(tasks),'intent_calls':2*len(tasks),
              'helpfulness_calls':240,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD,CASES,SOURCE/'questions_clean.parquet',SOURCE/'answers_clean.parquet']},
              'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.json')},
              'sampling_cells':dict(collections.Counter(r['year_band']+'/'+r['code_band']+'/'+r['score_band'] for r in selected))}
    dump('manifest.json',manifest)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
