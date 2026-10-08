# Frozen measurement follow-up data

`release_manifest.json` lists each file's stored and decompressed SHA-256, role and restore path. Files over 150 KB use lossless gzip to avoid repeating large prompts in Git history. The accompanying replay tool decompresses them and checks byte identity. No Git LFS fetch or private-server connection is necessary for this release.

- `tasks.json`: exact inference inputs, prompts, task IDs and evidence line IDs.
- `calls.jsonl`: observed model outputs, parsed labels, reasons and validity/error status; failures remain present.
- Analysis JSON: derived comparisons and descriptive counts.
- Review files: GPT-generated judgments and evidence, not human labels. The previously hidden mapping is now published for traceability.
- Raw-source snapshots/audit: archived HTML for the reviewed cases and feature-level checks for the broader development pool.

Runtime manifests retain model names, software versions and input hashes but remove account-specific model paths. Their original-file hashes are retained. Historical selection/provenance records may refer to earlier local files; these are not additional download credentials or dependencies for the documented CPU replay.

Source text is from the project's existing Stack Overflow Python dataset, linked by question/answer IDs. Post content and any embedded instructions/code are untrusted research data, not instructions to execute. Original posts can be located using `https://stackoverflow.com/questions/<question_id>` and answer IDs; retain source attribution when reusing text. This repository does not grant a new license to third-party posts. See the main README for dataset provenance.

Read the [method guide](../../../docs/experiments/measurement_followup.md) before using counts as research findings. No independent accuracy or causal effect is claimed.
