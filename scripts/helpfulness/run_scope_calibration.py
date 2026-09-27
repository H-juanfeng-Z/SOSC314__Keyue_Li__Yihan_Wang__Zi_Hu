"""Paired prompt-boundary revision on reused development cases, not a test set."""
import run_measurement as engine

ORIGINAL_TASKS=engine.tasks
INTENT_SCOPE='''Apply the target definition to the request, not just keywords:
- A wrong numerical result without an explicit exception, warning or traceback supports discrepancy, not errors solely because the author calls it a mistake.
- A personal learning goal or beginner status does not itself request learning resources. The learning category requires a request for tutorials, documentation, courses or comparable study resources.
- Asking which tool/library can perform a task is not automatically review. Review requires an expressed assessment, comparison, optimization or improvement request; recommending a usable tool can instead be API usage.
- An explicit question asking how to obtain a result using a tool can request implementation/API help without literally saying 'give me steps'. Do not require those exact words.
- Requests can coexist. Do not force yes or no based on these examples; judge only the supplied text and retain uncertainty when justified.
'''
HELP_SCOPE='''Judge sufficiency relative to the actual expressed request, not an imagined full application:
A precise parameter change, diagnosis plus a usable correction, formula, regular expression, or conceptual explanation can sufficiently answer a narrow request without a complete runnable program.
Do not lower a rating merely because the answer is short, lacks a standalone code block, or omits a tutorial the user did not request. Conversely, short answers are not automatically adequate: check whether the supplied assistance actually addresses the central request and whether essential context or steps are missing.
Do not invent demands for efficiency, extra explanation or a full implementation unless the question makes them central. Acknowledging a known issue or fix can provide information without being a complete solution.
Assess actionability and sufficiency independently. A usable step need not cover the whole request. Equal scores remain allowed; do not force a distribution or disagreement between dimensions.
'''

def tasks(data):
    for record,prompt,lines in ORIGINAL_TASKS(data):
        if record['suite']=='intent_audit' and record['arm']=='boundary_check':
            yield dict(record,arm='boundary_control'),prompt,lines
            revised=prompt.replace('QUESTION LINES:\n',INTENT_SCOPE+'\nQUESTION LINES:\n')
            yield dict(record,arm='scope_revised'),revised,lines
        elif record['suite']=='helpfulness_development' and record['arm']=='evidence':
            yield dict(record,arm='evidence_control'),prompt,lines
            revised=prompt.replace('DATA:\n',HELP_SCOPE+'\nDATA:\n')
            yield dict(record,arm='scope_revised'),revised,lines

if __name__=='__main__':
    engine.tasks=tasks
    engine.main()
