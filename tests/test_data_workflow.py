"""Small interface tests only: synthetic rows, no posts/models/network."""
import csv,importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def run(script,*args,cwd=ROOT):
 c=subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=cwd,capture_output=True,text=True,encoding='utf8',env=dict(os.environ,PYTHONUTF8='1',MPLBACKEND='Agg'),timeout=90)
 if c.returncode:raise AssertionError(str(script)+'\n'+c.stdout+'\n'+c.stderr)
 return c
class Workflow(unittest.TestCase):
 def test_end_to_end_synthetic_csv(self):
  import duckdb
  with tempfile.TemporaryDirectory(prefix='sosc314-smoke-') as td:
   t=Path(td);raw=t/'raw';raw.mkdir()
   def csvfile(name,header,rows):
    with (raw/name).open('w',encoding='utf8',newline='') as f:w=csv.writer(f);w.writerow(header);w.writerows(rows)
   csvfile('Questions.csv',['Id','OwnerUserId','CreationDate','Score','Title','Body'],[
    [1,10,'2014-01-01T00:00:00Z',0,'How to transform a list?','<p>Need an array.</p><pre><code>x=[1]</code></pre>'],
    [2,11,'2015-01-01T00:00:00Z',1,'Configure a server?','<p>How to configure a server?</p>'],
    [3,12,'2016-01-01T00:00:00Z',2,'Unanswered test','<p>No answer.</p>']])
   csvfile('Answers.csv',['Id','OwnerUserId','CreationDate','ParentId','Score','Body'],[
    [101,20,'2014-01-01T00:10:00Z',1,1,'<p>Use this function.</p>'],
    [102,21,'2014-01-01T00:20:00Z',1,2,'<p>Another approach.</p>'],
    [201,22,'2015-01-01T00:40:00Z',2,1,'<p>Set the option.</p>'],
    [202,23,'2014-12-31T23:59:00Z',2,0,'<p>Time inversion fixture.</p>']])
   csvfile('Tags.csv',['Id','Tag'],[[1,'python'],[1,'numpy'],[2,'python'],[3,'python'],[999,'orphan']])
   run('scripts/inspect_raw_data.py','--raw-dir',raw,'--out',t/'inventory.json')
   self.assertEqual(json.loads((t/'inventory.json').read_text())['files']['Questions.csv']['data_rows'],3)
   run('scripts/simple_exclusion.ipynb',cwd=raw)
   features=t/'features.parquet';normalized=t/'normalized'
   run('scripts/02_build_question_features.py','--raw-dir',raw,'--exclusions',raw/'excluded_records.csv','--working-dir',normalized,'--output',features,'--summary',t/'summary.csv')
   con=duckdb.connect()
   rows=con.execute('SELECT question_id,answer_count,first_answer_id,first_response_minutes FROM read_parquet(?) ORDER BY question_id',[str(features)]).fetchall()
   self.assertEqual(rows,[(1,2,101,10.0),(2,1,201,40.0),(3,0,None,None)])
   run('scripts/03_make_figures.py','--features',features,'--tables-dir',t/'tables','--figures-dir',t/'figures')
   self.assertEqual(len(list((t/'figures').glob('*.png'))),3)
   dest=t/'annotation'
   run('scripts/export_annotation_inputs.py','--features',features,'--normalized-dir',normalized,'--exclusions',raw/'excluded_records.csv','--out-dir',dest)
   manifest=json.loads((dest/'reconstruction_manifest.json').read_text());self.assertEqual((manifest['questions'],manifest['answers']),(3,3))
   qcols=[r[0] for r in con.execute('DESCRIBE SELECT * FROM read_parquet(?)',[str(dest/'questions_clean.parquet')]).fetchall()]
   self.assertTrue({'question_id','creation_year','title','body_html','question_created_at'}<=set(qcols))
   self.assertEqual(con.execute('SELECT answer_id FROM read_parquet(?) ORDER BY answer_id',[str(dest/'answers_clean.parquet')]).fetchall(),[(101,),(102,),(201,)])
   con.close()
   run('scripts/Feature Extract/03_extract_features_full.py','--parquet',dest/'questions_clean.parquet','--out',t/'features.xlsx')
   self.assertTrue((t/'features.xlsx').is_file())
 def test_recorded_manifest_sections(self):
  builder=load('manifest_builder',ROOT/'scripts/02_build_question_features.py')
  answers,tags=builder.read_exclusion_manifest(ROOT/'excluded_records.csv')
  self.assertEqual((len(answers),len(tags)),(25,4))
 def test_feature_compatibility_import(self):
  path=ROOT/'scripts/Feature Extract';sys.path.insert(0,str(path))
  try:
   m=load('feature_compat',path/'feature_extract_singlequestion.py')
   values=m.extract_features('Example?','<p>Error: ValueError</p><pre><code>x=1</code></pre>')
   self.assertIsInstance(values,dict);self.assertGreater(len(values),5)
  finally:sys.path.remove(str(path))
if __name__=='__main__':unittest.main()
