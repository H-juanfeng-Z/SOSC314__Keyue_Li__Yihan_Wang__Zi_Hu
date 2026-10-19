"""Export schema-compatible body inputs after Stage 2; no annotations/inference.

This new portability bridge does not reproduce historical Parquet bytes or sample
order. It applies the existing exclusion manifest without changing its rules.
"""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import duckdb
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--features',type=Path,default=Path('data/processed/question_response_features.parquet'))
 p.add_argument('--normalized-dir',type=Path,default=Path('data/interim/normalized_utf8'))
 p.add_argument('--exclusions',type=Path,default=Path('data/raw_csv/excluded_records.csv'))
 p.add_argument('--out-dir',type=Path,default=Path('data/processed/annotation_inputs'))
 a=p.parse_args();inputs=[a.features,a.normalized_dir/'Answers.csv',a.exclusions]
 assert all(x.is_file() for x in inputs),'Run screening and Stage 2 first'
 if a.out_dir.exists():raise FileExistsError('Use a NEW output directory; existing data are never replaced')
 a.out_dir.mkdir(parents=True)
 spec=importlib.util.spec_from_file_location('feature_builder',Path(__file__).with_name('02_build_question_features.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 excluded,_=mod.read_exclusion_manifest(a.exclusions)
 con=duckdb.connect();con.execute('SET TimeZone=\'UTC\'');con.register('excluded_answers',excluded)
 con.execute('CREATE TEMP TABLE q AS SELECT * FROM read_parquet(?)',[str(a.features)])
 con.execute("""CREATE TEMP TABLE a AS SELECT CAST(a.ParentId AS BIGINT) AS question_id,
 CAST(a.Id AS BIGINT) AS answer_id, TRY_CAST(a.OwnerUserId AS BIGINT) AS owner_user_id,
 a.Body AS body_html, CAST(a.CreationDate AS TIMESTAMPTZ) AS answer_created_at,
 CAST(a.Score AS INTEGER) AS answer_score
 FROM read_csv_auto(?,header=true,all_varchar=true,parallel=false) a
 JOIN q ON CAST(a.ParentId AS BIGINT)=q.question_id
 LEFT JOIN excluded_answers x ON CAST(a.Id AS BIGINT)=x.answer_id
 WHERE x.answer_id IS NULL""",[str(a.normalized_dir/'Answers.csv')])
 assert con.execute('SELECT count(*) FROM a JOIN q USING(question_id) WHERE answer_created_at<question_created_at').fetchone()[0]==0
 assert con.execute('SELECT count(*)=count(DISTINCT answer_id) FROM a').fetchone()[0]
 assert con.execute('SELECT count(*)=count(DISTINCT question_id) FROM q').fetchone()[0]
 for name,sql in [('questions_clean.parquet','SELECT * FROM q ORDER BY question_id'),('answers_clean.parquet','SELECT * FROM a ORDER BY question_id,answer_created_at,answer_id')]:
  con.execute('COPY ('+sql+') TO ? (FORMAT PARQUET, COMPRESSION ZSTD)',[str(a.out_dir/name)])
 sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
 result={'purpose':'Schema-compatible reconstruction, NOT historical-byte or historical-selection reproduction','inputs':{str(f):sha(f) for f in inputs},'files':{n:sha(a.out_dir/n) for n in ['questions_clean.parquet','answers_clean.parquet']},'questions':con.execute('SELECT count(*) FROM q').fetchone()[0],'answers':con.execute('SELECT count(*) FROM a').fetchone()[0],'new_scores_or_labels':False,'ordering':'question_id; answers then answer_created_at,answer_id'}
 (a.out_dir/'reconstruction_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf8');con.close();print(json.dumps(result,indent=2))
if __name__=='__main__':main()
