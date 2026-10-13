统一首次回答评分数据

1. scores_0_2_5000.json
   5000 个不同问题，每题一行首次回答的冻结 0–2 评分。
   来自 expanded_v1 的 2000 题、oct12 的 1000 题、oct13 的 2000 题。
   不含 pilot390、后续回答、控制任务或重复评分版本。

2. scores_0_10_1000.json
   上述5000题中，oct13固定子集的1000题，每题一行。
   format 和 scope 两个对象分别保存该候选版本的 score、status、usable。
   两个版本不平均、不择高，不自动替换0–2评分。

3. scores_0_2_matched_1000.json
   与第二个文件完全相同的1000个 question_id 和 answer_id，顺序也一致。
   内容是第一个文件的原样子集，不是额外1000题，不要重复追加。

使用方法
按 question_id 连接技术主题或意图分类。按 question_id + answer_id 比较评分。
0–2用 overall_numeric；相关性、可操作性、充分性分别为 relevance/actionability/sufficiency。
只把 usable=true 的记录用于对应分数分析。invalid、跳过、null保留，不填成0。
比较量表时取所比较版本共同 usable 的问题，报告实际分母。
batch 保留批次来源。这里整理的是评分，并不保证伙伴分类覆盖全部5000题。
模型评分不是人工真值。原来源许可和署名要求仍适用。

manifest.json记录数量和SHA256。python verify.py可离线核验。
旧文件和旧路径保持不变。本目录未重新评分或修改原始标签。
