"""Hash-sampled new development pool; exclude all earlier/reserved cases."""
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

def main():
    root=WORK/'sosc314_week4_experiments/request_bridge_20260927';out=root/'data';out.mkdir(exist_ok=True)
    sources=[WORK/'sosc314_week4_experiments/data/blinded_pairs.json',
             WORK/'sosc314_week4_experiments/measurement_20260926/source_cases.json',
             WORK/'sosc314_week4_experiments/measurement_20260926/data/helpfulness_development120.json']
    excluded=set()
    for p in sources:
        excluded.update(int(r['question_id']) for r in json.loads(p.read_text(encoding='utf8')) if r.get('question_id') is not None)
    con=duckdb.connect();con.execute("SET threads=4");con.execute("SET memory_limit='2GB'")
    cleaned=WORK/'sosc314-stackoverflow/data/processed'
    for name,view in [('questions_clean.parquet','q'),('answers_clean.parquet','a')]:con.read_parquet(str(cleaned/name)).create_view(view)
    con.execute('CREATE TEMP TABLE excluded(question_id BIGINT)')
    con.executemany('INSERT INTO excluded VALUES (?)',[(i,) for i in sorted(excluded)])
    cursor=con.execute('''WITH first_answer AS (
      SELECT * EXCLUDE(rn) FROM (SELECT *,row_number() OVER(PARTITION BY question_id ORDER BY answer_created_at,answer_id) rn FROM a) WHERE rn=1
    ) SELECT q.question_id,a.answer_id,q.title,q.body_html question_html,a.body_html answer_html,
      q.creation_year,a.answer_score,CAST(q.question_created_at AS VARCHAR) question_created_at,
      CAST(a.answer_created_at AS VARCHAR) answer_created_at,
      count(*) OVER() eligible_population
      FROM q JOIN first_answer a USING(question_id)
      WHERE q.creation_year BETWEEN 2008 AND 2016 AND q.question_id NOT IN(SELECT question_id FROM excluded)
        AND a.answer_created_at>=q.question_created_at
        AND length(trim(coalesce(q.body_html,'')))>0 AND length(trim(coalesce(a.body_html,'')))>0
      ORDER BY md5(CAST(q.question_id AS VARCHAR)||':bridge:20260927:314'),q.question_id LIMIT 1200''')
    columns=[c[0] for c in cursor.description];rows=[dict(zip(columns,r)) for r in cursor.fetchall()]
    assert len(rows)==1200 and len({r['question_id'] for r in rows})==1200
    assert not ({r['question_id'] for r in rows}&excluded)
    blinded=[];metadata=[]
    for i,r in enumerate(rows):
        pid=f'B27-{i+1:04d}'
        q=readable(r['question_html']);answer=readable(r['answer_html'])
        blinded.append({'pair_id':pid,'question_id':r['question_id'],'answer_id':r['answer_id'],'title':r['title'],'question_text':q,'answer_text':answer})
        metadata.append({'pair_id':pid,'question_id':r['question_id'],'answer_id':r['answer_id'],'year':r['creation_year'],
            'answer_score':r['answer_score'],'question_created_at':r['question_created_at'],'answer_created_at':r['answer_created_at'],
            'question_chars':len(q),'answer_chars':len(answer),'question_marked_code':'[CODE]' in q,'answer_marked_code':'[CODE]' in answer})
    for name,value in [('pairs.json',blinded),('analysis_metadata_private.json',metadata)]:
        with (out/name).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    manifest={'primary_pairs':200,'pool_pairs':1200,'excluded_ids':len(excluded),'eligible_population':rows[0]['eligible_population'],
        'selection':'Deterministic hash order, not stratified or selected by scores; first200 primary; later pairs prespecified extensions',
        'reserved_used':False,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json')},
        'sources':{str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources+[cleaned/'questions_clean.parquet',cleaned/'answers_clean.parquet']}}
    with (out/'manifest.json').open('x',encoding='utf8') as f:json.dump(manifest,f,indent=2)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
