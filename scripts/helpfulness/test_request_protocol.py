import json
import tempfile
import unittest
from pathlib import Path
import request_protocol as protocol
from summarize_bridge import summarize

class Tests(unittest.TestCase):
    def test_extraction_schema(self):
        obj={'request_summary':'How to configure a tool','request_evidence_ids':['T'],
             'facts':{k:{'present':'no','evidence_id':None} for k in protocol.FACT_KEYS}}
        self.assertEqual(protocol.validate_request(obj,{'T':'question'}),[])
        obj['facts']['implementation_request']={'present':'yes','evidence_id':'FAKE'}
        self.assertIn('implementation_request/evidence_id',protocol.validate_request(obj,{'T':'question'}))
    def test_missing_fact(self):
        self.assertIn('fact_keys',protocol.validate_request({'request_summary':'x','request_evidence_ids':['T'],'facts':{}},{'T':'question'}))
    def test_intent_uncertain(self):
        self.assertEqual(protocol.validate_intent({'decision':'uncertain','evidence_id':None,'reason':'Not explicit'},{'T':'q'}),[])
    def test_paired_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            rows=[{'pair_id':'one','suite':'helpfulness','arm':arm,'category':None,'sample_phase':'primary','status':'valid',
                   'parsed':{'overall':rating,'actionability':1,'sufficiency':1}} for arm,rating in [('direct','partial'),('question_first','substantial')]]
            (root/'calls.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf8')
            result=summarize(root)
            self.assertEqual(result['samples']['primary']['helpfulness_transitions']['partial -> substantial'],1)
            self.assertEqual(result['samples']['primary']['observed_pairs'],1)

if __name__=='__main__':unittest.main()
