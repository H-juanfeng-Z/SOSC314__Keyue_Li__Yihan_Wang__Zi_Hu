# Frozen intent expansion program (results pending)

Question-only Beyer seven-intent protocol: api_change,api_usage,conceptual,
discrepancy,errors,learning,review. Each question receives seven independent
yes/no/uncertain decisions; categories can overlap. No category is mandatory.
Model labels are provisional, not human ground truth. See prompt_templates.json
and request_protocol.py for operational definitions and strict evidence rules.

Prepare from the existing frozen local inputs:
`python scripts/intent_annotation/expansion_oct11/prepare.py --questions /path/to/questions.json --previous-tasks /path/to/old400/tasks.json --out /new/task-directory`.
Question records require question_id,title,body_text,base_pair_id. The original
question-only task file supplies the unchanged seven prompt templates. These
private inference inputs are not bundled here; this is code publication, not a
claim that a fresh clone contains the full input corpus.

Fixed expansion:1,600 previously unlabelled development questions from the frozen
2,000-question score release, four400-question shards. Each shard has2repeated
questions×7controls:2,814 tasks,11,256 total. Existing400 results are retained,
including failures. This commit contains NO complete expansion results and does
not replace any dataset release. No reserved80 used and no helpfulness rerating.

Run only after checking allocated GPU resources:
`python scripts/intent_annotation/expansion_oct11/run.py --data /path/to/tasks.json
--model /path/to/Qwen2.5-Coder-14B-Instruct --out /new/output`.
Requires PyTorch,transformers,safetensors and existing local model weights.
Input is a JSON list with task_id,pair_id,question_id,suite,arm='direct',
validator='intent',lines:{T:title,Q1:question_line,...},messages:[system,user].
Append JSON-serialized lines to the matching template's user_prefix followed by
QUESTION LINES:\n. Do not provide answers, scores or timing to this classifier.
The runner records hashes, keeps invalid/skipped outputs, refuses overwrite,
uses greedy256-token outputs and skips inputs above7,500tokens without truncation.
Slurm wrappers, connection programs and private host paths are not published.
