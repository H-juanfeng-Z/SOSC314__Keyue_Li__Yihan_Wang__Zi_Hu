"""Validate the exact release bytes using only Python's standard library."""
from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parent
manifest=json.loads((R/'release_manifest.json').read_text(encoding='utf8'))
for name,expected in manifest['files'].items():
    p=(R/name).resolve();assert p.is_relative_to(R.resolve()) and p.is_file(),name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
html=(R/'index.html').read_text(encoding='utf8')
assert len(re.findall(r'<img[^>]+src="data:image/png;base64,',html))==7
assert 'human gold' not in html.lower()
nb=json.loads((R/'report.ipynb').read_text(encoding='utf8'))
assert all(c.get('execution_count') is not None and all(o['output_type']!='error' for o in c['outputs']) for c in nb['cells'] if c['cell_type']=='code')
assert 'The authors also manually inspected' in ''.join(''.join(c['source']) for c in nb['cells'])
assert (R/'qwen_prompts.json').is_file() and (R/'qwen_example.json').is_file()
print('Release hashes, executed notebook,7 embedded figures and manual-review statement verified.')
