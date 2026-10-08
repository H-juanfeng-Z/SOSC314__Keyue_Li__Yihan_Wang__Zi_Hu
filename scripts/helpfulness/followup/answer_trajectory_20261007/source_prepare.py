"""First five observed answers for existing390 development questions; blinded rating."""
import sys,json,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parents[1]
sys.path.insert(0,str(W/'upload_preparation/python_deps'));sys.path.insert(0,str(R.parent))
import duckdb
from prepare_helpfulness import readable
def save(n,x):
 with (R/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,default=str)
pool=json.loads((R.parent/'association_sensitivity_20261007/input.json').read_text(encoding='utf8'));byq={p['question_id']:p for p in pool}
c=duckdb.connect();c.execute("SET threads=2");c.execute("SET memory_limit='2GB'");c.execute("SET TimeZone='UTC'")
for name,stamp in [('questions','question_created_at'),('answers','answer_created_at')]:
 c.read_parquet(str(W/f'sosc314-stackoverflow/data/processed/{name}_clean.parquet')).project(f'* REPLACE(CAST({stamp} AS TIMESTAMP) AS {stamp})').create_view(name)
c.execute('CREATE TEMP TABLE selected(question_id BIGINT)');c.executemany('INSERT INTO selected VALUES (?)',[(q,) for q in byq])
cur=c.execute('''SELECT q.*, count(a.answer_id) answer_count FROM questions q JOIN selected s USING(question_id) LEFT JOIN answers a USING(question_id) GROUP BY ALL''');qs=[dict(zip([d[0] for d in cur.description],r)) for r in cur.fetchall()];qmap={q['question_id']:q for q in qs}
cur=c.execute('''SELECT a.*, row_number() OVER(PARTITION BY question_id ORDER BY answer_created_at,answer_id) answer_rank FROM answers a JOIN selected s USING(question_id) QUALIFY answer_rank<=5''');ans=[dict(zip([d[0] for d in cur.description],r)) for r in cur.fetchall()]
template=next(t for t in json.loads((R.parent/'rubric_check_20261007/tasks.json').read_text(encoding='utf8')) if t['suite']=='original' and t['arm']=='no_examples_control');prefix=template['messages'][1]['content'].split('DATA:\n',1)[0]
tasks=[];meta=[];features=[];excluded=[]
for q in qs:
 text=readable(q['body_html']);features.append({'pair_id':byq[q['question_id']]['pair_id'],'question_id':q['question_id'],'phase':byq[q['question_id']]['phase'],'year':q['creation_year'],'answer_count':q['answer_count'],'truncated_after5':q['answer_count']>5,'title_chars':q['title_chars'],'body_html_chars':q['body_html_chars'],'readable_whitespace_tokens':len(text.split()),'code_block_count':q['code_block_count'],'link_count':q['link_count'],'question_mark_count':q['question_mark_count'],'question_created_at':q['question_created_at'],'question_author_known':q['owner_user_id'] is not None})
for a in ans:
 q=qmap[a['question_id']];bid=byq[a['question_id']]['pair_id'];pid=f"{bid}-A{a['answer_id']}"
 if a['answer_created_at'] is None or q['question_created_at'] is None or a['answer_created_at']<q['question_created_at']:
  excluded.append({'pair_id':pid,'reason':'missing_or_negative_timestamp'});continue
 text=readable(a['body_html']);lines={f'A{i+1}':s for i,s in enumerate(text.splitlines()) if s.strip()};d={'title':q['title'],'question':readable(q['body_html']),'answer_present':bool(lines),'ANSWER_LINES':lines}
 tasks.append({'task_id':f'trajectory/{pid}/no_examples','pair_id':pid,'question_id':q['question_id'],'suite':'trajectory','arm':'no_examples','validator':'joint','lines':lines,'messages':[template['messages'][0],{'role':'user','content':prefix+'DATA:\n'+json.dumps(d,ensure_ascii=False)}]})
 meta.append({'pair_id':pid,'base_pair_id':bid,'question_id':q['question_id'],'answer_id':a['answer_id'],'answer_rank':a['answer_rank'],'answer_count':q['answer_count'],'self_answer':None if q['owner_user_id'] is None or a['owner_user_id'] is None else q['owner_user_id']==a['owner_user_id'],'created_at':a['answer_created_at'],'delay_seconds':(a['answer_created_at']-q['question_created_at']).total_seconds()})
tasks.sort(key=lambda t:(hashlib.sha256((str(t['question_id'])+':314').encode()).hexdigest(),t['pair_id']))
assert len({t['task_id'] for t in tasks})==len(tasks) and len(tasks)<=1950
save('tasks.json',tasks);save('answer_metadata.json',meta);save('question_features.json',features);save('excluded.json',excluded)
save('manifest.json',{'planned':len(tasks),'questions':len(qs),'truncated_questions':sum(q['answer_count']>5 for q in qs),'self_answers':sum(m['self_answer'] is True for m in meta),'author_unknown':sum(m['self_answer'] is None for m in meta),'tasks_sha256':hashlib.sha256((R/'tasks.json').read_bytes()).hexdigest(),'reserved_used':False})
save('plan.json',{'purpose':'Move from first-answer scoring to observed answer trajectories and self/other distinctions.','sample':'All390 existing development questions, first5 observed answers by timestamp and ID; not population representative or full discussions.','rating':'Fixed no-example Qwen rubric; author, votes, rank and timing hidden; answers independently judged with question, not earlier answers.','outcomes':['first answer versus any observed answer substantial','later improvement among questions with at least2 observed answers and valid first rating','self/other/unknown descriptive groups','descriptive delay to first observed substantial answer; no survival model without archive-censoring metadata'], 'missing':'Never code invalid/skipped as unhelpful. Report complete and partial observed trajectories separately.','limitations':['Answer text and question text may be edited snapshots','More answers create more opportunities for any-help success','No causal interpretation or verified correctness','No comment data, no complementary-answer synthesis','Five-answer cap and snapshot observation window','Self-answer unknown if either author ID missing'], 'features':'Simple observed presentation counts exported for later intent-adjusted association; not quality labels.', 'budget':'One GPU, four threads,3h cap. Preserve partial outputs, no automatic resume or Github upload.'})
for n in ['run.py','launch_newserver.py','request_protocol.py','run_helpfulness_grounding.py']:shutil.copyfile(R.parent/'rubric_check_20261007'/n,R/n)
print((R/'manifest.json').read_text())
