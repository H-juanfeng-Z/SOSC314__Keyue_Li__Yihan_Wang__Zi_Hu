import argparse,json,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();R=Path(__file__).resolve().parent
rows=[json.loads(s) for s in (a.root/'calls.jsonl').read_text(encoding='utf8').splitlines()];tasks={t['task_id']:t for t in json.loads((R/'tasks.json').read_text(encoding='utf8'))}
out={'records':len(rows),'unique_ids':len({r['task_id'] for r in rows}),'planned':len(tasks),'conditions':{},'transitions':{},'semantic_flags':[]}
idx={(r['pair_id'],r['suite'],r['arm']):r for r in rows}
for suite,arm in sorted({(r['suite'],r['arm']) for r in rows}):
 rs=[r for r in rows if r['suite']==suite and r['arm']==arm];v=[r for r in rs if r['status']=='valid'];key='overall' if suite=='helpfulness' else 'decision'
 out['conditions'][suite+'/'+arm]={'n':len(rs),'valid':len(v),'labels':dict(collections.Counter(r['parsed'][key] for r in v))}
 for r in v:
  if suite=='helpfulness':
   o=r['parsed'];ls=tasks[r['task_id']]['lines']
   if (not ls and (o['overall']!='unhelpful' or any(o[k]!=0 for k in ['relevance','actionability','sufficiency']))) or (o['overall'] in ['partial','substantial'] and not o['evidence_ids']):out['semantic_flags'].append(r['task_id'])
 for r in v:
  basearm='original' if suite=='helpfulness' else 'full';b=idx.get((r['pair_id'],suite,basearm))
  if arm!=basearm and b and b['status']=='valid':
   ct=out['transitions'].setdefault(suite+'/'+arm,{});k=b['parsed'][key]+' -> '+r['parsed'][key];ct[k]=ct.get(k,0)+1
prior=json.loads((R/'prior_controls.json').read_text(encoding='utf8'));same=valid=0
for r in prior:
 b=idx.get((r['pair_id'],'helpfulness','original'))
 if b and b['status']=='valid':valid+=1;same+=all(r['parsed'][k]==b['parsed'][k] for k in ['relevance','actionability','sufficiency','overall'])
out['help_control_reproduction']={'valid':valid,'same':same}
audit=json.loads((R/'audit.json').read_text(encoding='utf8'))
out['ablation_audit']={'no_citations':sum(not x['cited_ids'] for x in audit),'random_overlaps_cited':sum(x['overlap']>0 for x in audit),'all_lines_cited':sum(len(x['cited_ids'])==x['line_count'] for x in audit)}
out['ablation_subgroups']={}
for group,subset in [('nonoverlap_nonempty_citations',[x for x in audit if x['cited_ids'] and x['overlap']==0]),('overlapping_deletions',[x for x in audit if x['overlap']>0])]:
 counts=collections.Counter()
 for x in subset:
  c=idx.get((x['pair_id'],'helpfulness','cited_removed'));r=idx.get((x['pair_id'],'helpfulness','random_removed'))
  if c and r and c['status']==r['status']=='valid':counts[c['parsed']['overall']+' vs '+r['parsed']['overall']]+=1
 out['ablation_subgroups'][group]={'planned_pairs':len(subset),'paired_labels':dict(counts)}
(a.root/'analysis.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
