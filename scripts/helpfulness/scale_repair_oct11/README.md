# Candidate scale repair: diagnostic, not adopted

`python scripts/helpfulness/scale_repair_oct11/verify.py` verifies the public
242-task decision export, original schema, frozen summaries and26 prior controls.
The natural post bodies/raw completions remain local. This is a decision-export
audit, NOT byte-for-byte raw-log reproduction. Message hashes preserve provenance.
AI-constructed source cases are included separately; they are not human gold.

100 natural question/first-answer pairs (18previously reviewed+82additional reused
development cases) and8 synthetic cases receive coarse0–2/fine0–10 judgments;
26 exact historical fine controls make242 tasks. Controls are deterministic
reproduction, not inter-rater reliability. The reserved80 were not used.
Candidate system prompts separate assessor instructions from post requests and
clarify short answers/null. Role separation and wording changed together, so this
is not a pure scale-resolution experiment. Original labels are not replaced.

236/242 pass schema validation: four empty remaining_gap failures and two over-
length evidence lists. Preserve failures. Eighteen otherwise valid repaired
outputs exceed the100-word instruction, checked separately from schema validity.
No obvious assessor-instruction copying remained in216 request fields; this is
NOT full request-interpretation accuracy. Complete-code2 versus partial-code3
and invented implementation/confirmation demands remain. Wider spread is not
evidence of better measurement. Do not deploy this candidate over the full dataset.

Optional GPU runner: `python scripts/helpfulness/scale_repair_oct11/run.py
--data /path/to/private_tasks.json --model /path/to/local/model --out /new/output`.
It expects task_id,pair_id,question_id,suite,arm,lines,messages; arm coarse/fine/
fine_control only. Published tasks.json is an audit export, not inference input.
Requires PyTorch,transformers,safetensors and an already available model; no model
download or automatic job submission. The runner retains original limits and
validators; server-specific launch/connection code is deliberately excluded.
