"""Validate ratings and freeze them before opening the answer-order key."""
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
R = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf8'))
packet = {p['case_id']: p for p in read(R/'review_packet.json')}
files = sorted(R.glob('review_0[1-5].json'))
ratings = [r for f in files for r in read(f)]
assert len(ratings) == 31 and len({r['case_id'] for r in ratings}) == 31
assert {r['case_id'] for r in ratings} == set(packet)
for r in ratings:
    assert set(r['answers']) == {'X','Y'}
    for letter, v in r['answers'].items():
        assert v['label'] in {'substantial','partial','unhelpful','uncertain'}
        assert v['confidence'] in {'high','medium','low'}
        assert v['reason'] and v['evidence_ids']
        assert set(v['evidence_ids']) <= set(packet[r['case_id']]['answers'][letter]), (r['case_id'],letter,v['evidence_ids'])
tracked = files + [R/'review_packet.json', R/'protocol.json']
receipt = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(), cases=31, answers=62,
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
    statement='Ratings frozen before unmasking. AI evidence review, not human gold, not strict blind; no code executed.')
with (R/'freeze_receipt.json').open('x',encoding='utf8') as f: json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
