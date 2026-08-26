# Failure Cluster Analysis — Phase A

**Evaluation backend:** `ragas`
**Questions evaluated:** 50

## Worst-metric × distribution matrix

| Metric | factual | multi_hop | adversarial | total |
|---|---:|---:|---:|---:|
| faithfulness | 1 | 8 | 0 | 9 |
| answer_relevancy | 17 | 8 | 2 | 27 |
| context_precision | 2 | 0 | 0 | 2 |
| context_recall | 0 | 4 | 8 | 12 |

## Dominant failure

- Distribution: **factual**
- Metric: **answer_relevancy**
- Insight: 'factual' contains the largest failure cluster and 'answer_relevancy' is the most frequent weakest metric. Recommended next action: Preserve intent/entities/units and make the generation prompt more direct.

## Bottom 10

| Rank | QID | Distribution | Avg score | Worst metric | Diagnosis | Suggested fix |
|---:|---:|---|---:|---|---|---|
| 1 | 33 | multi_hop | 0.500 | answer_relevancy | The answer does not directly satisfy the question. | Preserve intent/entities/units and make the generation prompt more direct. |
| 2 | 24 | multi_hop | 0.625 | faithfulness | The generated answer is not sufficiently grounded in retrieved evidence. | Use stricter context-only generation, remove noisy chunks, and keep temperature low. |
| 3 | 38 | multi_hop | 0.704 | faithfulness | The generated answer is not sufficiently grounded in retrieved evidence. | Use stricter context-only generation, remove noisy chunks, and keep temperature low. |
| 4 | 23 | multi_hop | 0.729 | faithfulness | The generated answer is not sufficiently grounded in retrieved evidence. | Use stricter context-only generation, remove noisy chunks, and keep temperature low. |
| 5 | 29 | multi_hop | 0.744 | faithfulness | The generated answer is not sufficiently grounded in retrieved evidence. | Use stricter context-only generation, remove noisy chunks, and keep temperature low. |
| 6 | 25 | multi_hop | 0.785 | faithfulness | The generated answer is not sufficiently grounded in retrieved evidence. | Use stricter context-only generation, remove noisy chunks, and keep temperature low. |
| 7 | 7 | factual | 0.797 | faithfulness | The generated answer is not sufficiently grounded in retrieved evidence. | Use stricter context-only generation, remove noisy chunks, and keep temperature low. |
| 8 | 27 | multi_hop | 0.803 | faithfulness | The generated answer is not sufficiently grounded in retrieved evidence. | Use stricter context-only generation, remove noisy chunks, and keep temperature low. |
| 9 | 45 | adversarial | 0.831 | context_recall | The retrieved context is missing required evidence. | Improve chunking, hybrid recall, query decomposition, or parent retrieval. |
| 10 | 26 | multi_hop | 0.832 | faithfulness | The generated answer is not sufficiently grounded in retrieved evidence. | Use stricter context-only generation, remove noisy chunks, and keep temperature low. |
