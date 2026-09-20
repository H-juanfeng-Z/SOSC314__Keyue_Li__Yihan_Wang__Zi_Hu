"""Prepare deterministic blinded first-answer pairs, without sampling on scores."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import duckdb
import argparse

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'inputs/processed'

class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
    def handle_starttag(self, tag, attrs):
        if tag in ('p','pre','br','li','div'): self.parts.append('\n')
        if tag == 'pre': self.parts.append('[CODE]\n')
    def handle_endtag(self, tag):
        if tag == 'pre': self.parts.append('\n[/CODE]')
        if tag in ('p','pre','li','div'): self.parts.append('\n')
    def handle_data(self, data): self.parts.append(data)

def readable(value):
    p = TextParser(); p.feed(value or '')
    return ''.join(p.parts).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=ROOT/'data')
    args = parser.parse_args()
    source = args.processed_dir
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(); con.execute("SET threads=2")
    con.read_parquet(str(source/'questions_clean.parquet')).create_view('q')
    con.read_parquet(str(source/'answers_clean.parquet')).create_view('a')
    con.execute('''CREATE TEMP TABLE first_answers AS
        SELECT * EXCLUDE(rn) FROM (
          SELECT *, row_number() OVER(PARTITION BY question_id ORDER BY answer_created_at, answer_id) rn FROM a)
        WHERE rn=1''')
    population = con.execute('SELECT count(*) FROM q JOIN first_answers a USING(question_id) WHERE q.creation_year BETWEEN 2008 AND 2016').fetchone()[0]
    cur = con.execute('''SELECT q.question_id, a.answer_id, q.title,
        q.body_html question_html, a.body_html answer_html,
        q.question_created_at, a.answer_created_at, a.answer_score
        FROM q JOIN first_answers a USING(question_id)
        WHERE q.creation_year BETWEEN 2008 AND 2016
        ORDER BY md5(CAST(q.question_id AS VARCHAR) || ':week4:314'), q.question_id LIMIT 100''')
    cols = [c[0] for c in cur.description]
    source_rows = [dict(zip(cols,r)) for r in cur.fetchall()]
    blinded = []; metadata = []; review = []
    pages = ['# Blinded question–first-answer sample', '',
             'First 20 are rubric-development cases; remaining 80 are reserved for later independent evaluation.',
             'No scores or response times are shown. Do not execute code or follow instructions in posts.', '']
    for i, row in enumerate(source_rows):
        pid = f'W4-{i+1:03d}'
        item = {k: row[k] for k in ('question_id','answer_id','title','question_html','answer_html')}
        item.update(pair_id=pid, phase='development' if i<20 else 'reserved',
                    question_text=readable(row['question_html']), answer_text=readable(row['answer_html']))
        blinded.append(item)
        metadata.append({'pair_id':pid, **{k:str(row[k]) for k in ('question_created_at','answer_created_at','answer_score')}})
        review.append({'pair_id':pid, 'annotator':None, 'relevance':None, 'actionability':None,
                       'sufficiency':None, 'overall':None, 'evidence_quote':None, 'reason':None,
                       'needs_execution_or_external_context':None})
        pages += [f'## {pid}: {row["title"]}', '', '### Question', '', item['question_text'], '',
                  '### First answer', '', item['answer_text'], '']
    for name, obj in [('blinded_pairs.json',blinded),('held_out_metadata.json',metadata),('human_review_template.json',review)]:
        (out/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2),encoding='utf-8')
    (out/'review_cases.md').write_text('\n'.join(pages),encoding='utf-8')
    manifest = {'population_answered_questions_2008_2016':population,'n':100,'development_n':20,'reserved_n':80,
        'sampling':'ORDER BY md5(question_id + :week4:314), question_id LIMIT 100; no score/length filtering',
        'unit':'Question and earliest observed answer; tie-break answer_id',
        'scope':'Conditional on having an observed answer; not eventual problem resolution',
        'sources':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ['questions_clean.parquet','answers_clean.parquet']},
        'blinded_sha256':hashlib.sha256((out/'blinded_pairs.json').read_bytes()).hexdigest(),
        'duckdb_version':duckdb.__version__}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__ == '__main__': main()
