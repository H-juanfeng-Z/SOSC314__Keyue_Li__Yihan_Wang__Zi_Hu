import json,gzip,statistics,math,hashlib
from pathlib import Path
from collections import Counter
from bisect import bisect_left,bisect_right
R=Path(__file__).resolve().parents[3]
T=R/'tables/response_timing/oct11'
d=json.loads(gzip.decompress((T/'joined_data.json.gz').read_bytes()))
expected=json.loads((T/'analysis.json').read_text())
timing={r['question_id']:r for r in d['rows']}
scores={r['question_id']:{'scores':{'overall':r['overall']},'usable':r['usable']} for r in d['expanded_scores']}
qs=set(scores);groups=d['memberships'];complete=set.union(*(set(v) for v in groups['beyer'].values()))
def valid(q):return q in timing and timing[q]['seconds'] is not None and timing[q]['seconds']>=0
def usable(q):return q in scores and scores[q]['usable']
reference=sorted(timing[q]['seconds'] for q in qs if valid(q))
def weight(q,floor=.5):
 x=timing[q]['seconds'];rank=(bisect_left(reference,x)+1+bisect_right(reference,x))/2
 return 1-(1-floor)*(rank-1)/(len(reference)-1)
def quantile(v,p):
 z=sorted(v);i=(len(z)-1)*p;l=int(i);u=min(l+1,len(z)-1);return z[l]+(z[u]-z[l])*(i-l)
def summary(ids):
 ids=set(ids);good=sorted(q for q in ids if valid(q));v=[timing[q]['seconds']/60 for q in good]
 o=dict(selected_n=len(ids),source_missing_n=len(ids-set(timing)),no_observed_answer_n=sum(q in timing and timing[q]['answer_id'] is None for q in ids),negative_time_n=sum(q in timing and timing[q]['seconds'] is not None and timing[q]['seconds']<0 for q in ids),valid_answered_n=len(good))
 if not v:return o
 o.update(mean_minutes=statistics.mean(v),q25_minutes=quantile(v,.25),median_minutes=statistics.median(v),q75_minutes=quantile(v,.75),p90_minutes=quantile(v,.9),zero_delay_n=v.count(0),within_1h_n=sum(x<=60 for x in v),within_24h_n=sum(x<=1440 for x in v))
 o['within_1h_fraction_answered']=o['within_1h_n']/len(v);o['within_24h_fraction_answered']=o['within_24h_n']/len(v)
 matched=[q for q in good if usable(q)];o['matched_valid_score_n']=len(matched)
 if matched:
  levels={'unhelpful':0,'partial':1,'substantial':2};y=[levels[scores[q]['scores']['overall']] for q in matched]
  o['helpfulness_counts']=dict(Counter(scores[q]['scores']['overall'] for q in matched))
  o['substantial_fraction']=sum(x==2 for x in y)/len(y)
  o['exploratory_mean_ordinal012']=statistics.mean(y)
  o['weighted_index_means']={str(f):statistics.mean(levels[scores[q]['scores']['overall']]*weight(q,f) for q in matched) for f in [.25,.5,.75]}
  o['score_matched_timing_median_minutes']=statistics.median(timing[q]['seconds']/60 for q in matched)
 return o

actual={name:{k:summary(ids) for k,ids in g.items()} for name,g in groups.items()}
assert actual==expected['classification_tables']
assert summary(qs)==expected['overall_expanded2000']
delay={label:summary(q for q in qs if valid(q) and lo<timing[q]['seconds']/60<=hi) for label,lo,hi in [('0_to_1h',-1,60),('1_to_24h',60,1440),('over_24h',1440,float('inf'))]}
assert delay==expected['helpfulness_by_delay']
years={name:{str(y):{k:summary(q for q in ids if q in timing and timing[q]['creation_year']==y) for k,ids in g.items()} for y in sorted({timing[q]['creation_year'] for ids in g.values() for q in ids if q in timing})} for name,g in groups.items()}
assert years==expected['classification_by_year']
print(json.dumps({'passed':True,'group_tables':sum(map(len,actual.values())),'year_tables':sum(len(g) for y in years.values() for g in y.values()),'expanded_questions':len(qs),'scope':'Frozen joined-input replay, not raw archive extraction or label validation.'}))
