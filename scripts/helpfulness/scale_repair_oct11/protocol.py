def validate(o,t):
 if t['arm']=='legacy':
  from legacy_protocol import validate as old
  return old(o,dict(t,arm='baseline'))
 if not isinstance(o,dict):return ['not_object']
 errors=[];v=o.get('score');maximum=2 if t['arm']=='coarse' else 10
 if 'score' not in o or (v is not None and (type(v) is not int or not 0<=v<=maximum)):errors.append('score')
 for k in ['request','provided_help','remaining_gap','reason']:
  if not isinstance(o.get(k),str) or not o[k].strip():errors.append(k)
 if o.get('confidence') not in ['low','medium','high']:errors.append('confidence')
 ids=o.get('evidence_ids')
 if not isinstance(ids,list):errors.append('evidence_not_list')
 else:
  if len(ids)>3:errors.append('evidence_too_many')
  if any(not isinstance(i,str) or i not in t['lines'] for i in ids):errors.append('evidence_illegal_id')
 return errors
