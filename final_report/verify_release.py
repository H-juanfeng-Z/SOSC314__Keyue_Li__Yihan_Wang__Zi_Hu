from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parent
m=json.loads((R/'release_manifest.json').read_text(encoding='utf8'))
for name,h in m['files'].items():
    p=(R/name).resolve();assert p.is_relative_to(R.resolve()) and p.is_file();assert hashlib.sha256(p.read_bytes()).hexdigest()==h,name
n=json.loads((R/'report.ipynb').read_text(encoding='utf8'));assert all(c.get('execution_count') is not None and all(o['output_type']!='error' for o in c['outputs']) for c in n['cells'] if c['cell_type']=='code')
h=(R/'index.html').read_text(encoding='utf8');assert len(re.findall(r'<img[^>]+src="data:image/png;base64,',h))==11;assert 'human gold' not in h.lower();assert 'None detected' not in h
d=json.loads((R/'report_data.json').read_text(encoding='utf8'));assert sum(d['main_results']['coarse_counts'].values())==4986;assert sum(g['n'] for g in d['main_results']['lda_coarse_groups'].values())==4977;assert d['main_results']['timing_n']==1987
captions=re.findall(r'Figure (\d+)\.', '\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='markdown'))
assert captions==[str(i) for i in range(1,12)],captions
from recompute_code_element import summarize
support=json.loads((R/'code_element_support.json').read_text(encoding='utf8'))
summary=json.loads((R/'code_element_summary.json').read_text(encoding='utf8'))
assert summarize(support)==summary
assert summary['arms']['format']['valid_n']==995 and summary['arms']['scope']['valid_n']==996
assert all(len(re.findall(r'Figure '+str(i)+r'\.',h))==1 for i in range(1,12))
print('Notebook, hashes, 11 figures, preserved scores, code-element SEs and denominator checks passed.')
