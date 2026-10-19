"""Offline frozen-release checks; optionally replay the report. No model calls."""
import argparse,base64,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(rel):
 c=subprocess.run([sys.executable,str(ROOT/rel)],cwd=ROOT,text=True,capture_output=True,encoding='utf8')
 print(c.stdout,end='')
 if c.returncode:raise RuntimeError(rel+'\n'+c.stderr)
def pngs(n):
 return [hashlib.sha256(base64.b64decode(''.join(o['data']['image/png']))).hexdigest() for c in n['cells'] if c['cell_type']=='code' for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay-report',action='store_true');p.add_argument('--strict-images',action='store_true');a=p.parse_args()
 for script in ['final_report/verify_release.py','tables/reusable_evaluation/unified_v1/verify.py','tables/reusable_evaluation/oct13/verify.py','scripts/reusable_evaluation/verify_oct12.py','scripts/response_timing/oct11/reproduce.py','scripts/helpfulness/followup/reproduce.py']:
  run(script)
 if a.replay_report:
  run('final_report/reproduce.py')
  read=lambda p:json.loads(p.read_text(encoding='utf8'))
  before=read(ROOT/'final_report/report.ipynb');after=read(ROOT/'final_report/reproduced/report.ipynb')
  assert len(pngs(after))==11,'Missing displayed report images'
  exact=pngs(before)==pngs(after)
  if a.strict_images:assert exact,'PNG bytes differ; check versions, fonts and numerical outputs'
  assert [c['source'] for c in before['cells']]==[c['source'] for c in after['cells']]
  print(f'PASS: fresh report replay; 11 displayed images; exact PNG matches={exact}. See final_report/reproduced/index.html')
 print('PASS: frozen data/interface checks. These do not establish annotation accuracy or raw-inference reproduction.')
if __name__=='__main__':main()
