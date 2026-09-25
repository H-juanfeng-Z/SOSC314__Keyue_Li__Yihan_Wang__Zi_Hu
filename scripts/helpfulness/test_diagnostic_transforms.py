"""CPU-only tests: transformations and schema rules, not semantic correctness."""
import unittest
from run_content_sensitivity import transform
from run_code_grounding import trials
from run_helpfulness_grounding import validate

class Tests(unittest.TestCase):
    def test_removal(self):
        variants=dict(transform('Before\n[CODE]\nx=1\n[/CODE]\nAfter'))
        self.assertEqual(variants['prose_only'],'Before\n\nAfter')
        self.assertEqual(variants['code_only'],'x=1')
        self.assertEqual(len(variants['prefix_half']),(len(variants['original'])+1)//2)

    def test_swap(self):
        cases=[{'pair_id':'a','answer_text':'A[CODE]x=1[/CODE]B[CODE]y=2[/CODE]C'},
               {'pair_id':'b','answer_text':'D[CODE]z=3[/CODE]E'}]
        rows=list(trials(cases))
        self.assertEqual(len(rows),8)
        self.assertEqual(rows[1][2],'A[CODE]\nz=3\n[/CODE]BC')
        self.assertEqual(rows[1][3],'b')

    def test_schema(self):
        obj={'relevance':2,'actionability':1,'sufficiency':1,'overall':'partial','evidence_ids':['A1'],'reason':'Explanation'}
        self.assertEqual(validate(obj,{'A1':'text'}),[])
        self.assertIn('evidence_ids',validate(obj,{}))
        obj['relevance']=True
        self.assertIn('relevance',validate(obj,{'A1':'text'}))

if __name__=='__main__': unittest.main()
