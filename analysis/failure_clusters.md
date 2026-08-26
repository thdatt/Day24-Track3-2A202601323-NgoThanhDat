# Failure Cluster Analysis — Phase A

**Student:** Ngo Thanh Dat — 2A202601323  
**Evaluation backend:** `ragas`  
**Questions evaluated:** 50

---

## 1. Aggregate RAGAS Results

| Metric | Factual (20q) | Multi-hop (20q) | Adversarial (10q) | Weighted overall |
|---|---:|---:|---:|---:|
| Faithfulness | 0.975 | **0.695** | 1.000 | **0.868** |
| Answer relevancy | 0.841 | **0.751** | 0.856 | **0.808** |
| Context precision | 0.967 | 0.983 | 1.000 | **0.980** |
| Context recall | 1.000 | 0.879 | **0.717** | **0.895** |
| **Average score** | **0.946** | **0.827** | **0.893** | **0.888** |

### Key observation

The pipeline performs best on direct factual lookup. The weakest distribution by **normalized average score** is `multi_hop` (0.827), driven mainly by low faithfulness (0.695) and answer relevancy (0.751). Adversarial retrieval remains highly precise but has lower context recall (0.717), which indicates missing evidence rather than noisy retrieval.

---

## 2. Worst-Metric × Distribution Matrix

Each question contributes one count to the metric that was weakest for that question.

| Worst metric | Factual | Multi-hop | Adversarial | Total |
|---|---:|---:|---:|---:|
| Faithfulness | 1 | 8 | 0 | 9 |
| Answer relevancy | 17 | 8 | 2 | **27** |
| Context precision | 2 | 0 | 0 | 2 |
| Context recall | 0 | 4 | 8 | 12 |
| **Total** | **20** | **20** | **10** | **50** |

The generated cluster output labels `factual` as the dominant failure distribution and `answer_relevancy` as the dominant weakest metric. The metric conclusion is clear: **answer relevancy is the most frequent weakest dimension (27/50)**.

For distribution-level diagnosis, raw counts should be interpreted carefully because the test set contains different distribution sizes. The average-score view shows that `multi_hop` has the most important quality weakness despite factual questions contributing many answer-relevancy minima.

---

## 3. Bottom 10 Questions

| Rank | QID | Distribution | Avg. score | Worst metric | Diagnosis | Suggested fix |
|---:|---:|---|---:|---|---|---|
| 1 | 33 | multi_hop | 0.500 | answer_relevancy | The answer does not directly satisfy the question. | Preserve requested entities/units and answer the requested calculation directly. |
| 2 | 24 | multi_hop | 0.625 | faithfulness | Generated content is not sufficiently grounded in retrieved evidence. | Use stricter context-only generation and remove noisy evidence. |
| 3 | 38 | multi_hop | 0.704 | faithfulness | Generated content is not sufficiently grounded in retrieved evidence. | Decompose the question and ground each sub-answer in retrieved context. |
| 4 | 23 | multi_hop | 0.729 | faithfulness | Generated content is not sufficiently grounded in retrieved evidence. | Strengthen evidence-only prompting and source/version handling. |
| 5 | 29 | multi_hop | 0.744 | faithfulness | Generated content is not sufficiently grounded in retrieved evidence. | Retrieve evidence for every calculation/condition before generation. |
| 6 | 25 | multi_hop | 0.785 | faithfulness | Generated content is not sufficiently grounded in retrieved evidence. | Reduce unsupported inference and preserve exact policy conditions. |
| 7 | 7 | factual | 0.797 | faithfulness | The answer includes content that is not fully supported by the evidence. | Use a more restrictive grounded-answer prompt. |
| 8 | 27 | multi_hop | 0.803 | faithfulness | Generated content is not sufficiently grounded in retrieved evidence. | Improve multi-step retrieval and evidence aggregation. |
| 9 | 45 | adversarial | 0.831 | context_recall | Required evidence was not fully retrieved. | Improve hybrid recall, metadata/version filters and parent retrieval. |
| 10 | 26 | multi_hop | 0.832 | faithfulness | Generated content is not sufficiently grounded in retrieved evidence. | Require explicit evidence coverage before composing the final answer. |

Seven of the bottom ten cases are multi-hop questions, confirming that multi-step reasoning/grounding is the most important improvement area even though `answer_relevancy` is the most frequent weakest metric across all questions.

---

## 4. Failure Modes

### Failure Mode A — Multi-hop grounding loss

**Evidence:**
- Multi-hop faithfulness: **0.695**
- 8 multi-hop questions have faithfulness as their weakest metric.
- 7 of the bottom 10 examples are multi-hop.

**Likely mechanism:** the system may retrieve individually relevant chunks but lose a condition, arithmetic step, or cross-document relationship during answer synthesis.

**Improvement:**
1. Decompose multi-hop questions into explicit sub-queries.
2. Retrieve evidence per sub-query.
3. Preserve document/version metadata.
4. Build an evidence ledger before final generation.
5. Require each claim/calculation to be supported by retrieved context.

### Failure Mode B — Answer relevancy

**Evidence:** `answer_relevancy` is the weakest metric for **27/50** questions.

**Likely mechanism:** answers may be factually grounded but include more context than the user requested, omit an exact requested field, or fail to preserve entities/units/conditions precisely.

**Improvement:** use a direct-answer prompt with an explicit checklist: requested entity, requested condition, requested unit, requested calculation, and no unrelated policy details.

### Failure Mode C — Adversarial context recall

**Evidence:** adversarial context recall is **0.717**, the lowest context-recall score across the three distributions; 8 adversarial questions have context recall as their weakest metric.

**Likely mechanism:** version-conflict and contradiction questions need the system to retrieve both the tempting outdated/contradictory evidence and the authoritative current rule.

**Improvement:**
- filter/boost by document version and effective date;
- use hybrid lexical + dense search;
- retrieve parent sections around conflict-sensitive passages;
- explicitly rank current/authoritative policy above superseded versions.

---

## 5. Prioritized Remediation Plan

| Priority | Issue | Evidence | Proposed change | Success metric |
|---|---|---|---|---|
| P0 | Multi-hop faithfulness | 0.695 | Query decomposition + evidence-led generation | >= 0.75 |
| P1 | Answer relevancy | weakest in 27/50 | Direct-answer generation contract | overall >= 0.85 |
| P1 | Adversarial context recall | 0.717 | Version-aware retrieval + parent retrieval | >= 0.80 |
| P2 | Factual precision | 0.967 | Reduce unnecessary retrieved chunks | maintain >= 0.95 |

---

## 6. Conclusion

The RAG pipeline is strong overall (**weighted average score 0.888**) but the evaluation exposes two production-relevant weaknesses: multi-hop grounding and answer directness. The next iteration should not focus on increasing retrieval volume indiscriminately; it should improve evidence composition, version-aware retrieval, and concise intent-preserving generation.
