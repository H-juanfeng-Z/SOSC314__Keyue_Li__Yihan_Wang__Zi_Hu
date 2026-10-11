# Frozen scoring-diagnostic decision export

See [programs and reproduction instructions](../../../scripts/helpfulness/scale_repair_oct11/README.md).
`outputs.json` contains model-generated summaries and judgments, not original
natural post bodies. `tasks.json` contains identifiers, permitted evidence IDs
and hashes. `prior_fine_controls.json` supports the26control comparisons.
`protocol.json`/`prompts.json` preserve candidate definitions and system prompts.
`synthetic_cases.json` contains AI-constructed diagnostic inputs. No new scores
replace the released baseline. `review_findings.json` is GPT review, not human gold.

Natural-case identifiers refer to the historical Kaggle Python Questions from
Stack Overflow archive: https://www.kaggle.com/datasets/stackoverflow/pythonquestions.
Original question pages can be located at https://stackoverflow.com/questions/QUESTION_ID.
The export omits natural post bodies; consult original pages for authorship and
source context. Do not treat AI paraphrases as verbatim author quotations.
