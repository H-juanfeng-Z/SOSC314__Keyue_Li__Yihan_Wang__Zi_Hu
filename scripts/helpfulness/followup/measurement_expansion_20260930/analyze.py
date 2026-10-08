"""Paired diagnostic summaries; never label model agreement as accuracy."""
import argparse,collections,json
from pathlib import Path
def analyze(root,selection):
 rows=[json.loads(s) for s in (root/'calls.jsonl').read_text(encoding='utf8').splitlines()]
 old={r['task_id']:r for r in json.loads(selection.read_text(encoding='utf8'))}
 idx={(r['suite'],r['pair_id'],r['arm']):r for r in rows};assert len(idx)==len(rows)
 out={'records':len(rows),'statuses':dict(collections.Counter(r['status'] for r in rows)),'comparisons':{},'control_repetition':{}}
 def compare(suite,arm1,arm2,field1,field2):
  counts=collections.Counter();missing=0
  for pid in {r['pair_id'] for r in rows if r['suite']==suite}:
   a=idx.get((suite,pid,arm1));b=idx.get((suite,pid,arm2))
   if not a or not b or a['status']!='valid' or b['status']!='valid':missing+=1;continue
   group=old[a['task_id']]['selection_group']
   counts[group+'/'+str(a['parsed'][field1])+' -> '+str(b['parsed'][field2])]+=1
  out['comparisons'][suite+'/'+arm1+'/'+arm2]={'transitions':dict(counts),'invalid_or_missing':missing}
 compare('intent','direct_control','extraction_control','decision','decision')
 compare('intent','extraction_control','summary_only','decision','decision')
 compare('intent','direct_control','summary_only','decision','decision')
 compare('helpfulness','joint_control','joint_no_examples','overall','overall')
 compare('helpfulness','isolated_actionability','isolated_sufficiency','score','score')
 compare('helpfulness','joint_control','isolated_actionability','actionability','score')
 compare('helpfulness','joint_control','isolated_sufficiency','sufficiency','score')
 for arm in ['direct_control','extraction_control','joint_control']:
  valid=[r for r in rows if r['arm']==arm and r['status']=='valid']
  fields=['decision'] if arm!='joint_control' else ['relevance','actionability','sufficiency','overall']
  out['control_repetition'][arm]={'valid':len(valid),'same_decisions':sum(all(r['parsed'].get(k)==old[r['task_id']]['prior'].get(k) for k in fields) for r in valid)}
 for arm in ['joint_control','joint_no_examples']:
  valid=[r for r in rows if r['arm']==arm and r['status']=='valid'];numeric=[r for r in valid if all(type(r['parsed'][d]) is int for d in ['actionability','sufficiency'])]
  out[arm]={'valid':len(valid),'numeric':len(numeric),'dimensions_equal':sum(r['parsed']['actionability']==r['parsed']['sufficiency'] for r in numeric)}
 out['interpretation']='Selected development diagnostics, not semantic accuracy; altered isolated wording is a confound.'
 (root/'analysis.json').write_text(json.dumps(out,indent=2),encoding='utf8');return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);a=p.parse_args();print(json.dumps(analyze(a.root,a.selection),indent=2))
