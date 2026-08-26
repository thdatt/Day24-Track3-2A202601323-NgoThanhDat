# Lab 24 Final Report — Production Eval + Guardrail Stack

**Student:** Ngo Thanh Dat  
**Student ID:** 2A202601323  
**Project:** Day24-Track3-2A202601323-NgoThanhDat  
**Date:** 2026-08-26

---

## 1. Objective

The goal of Lab 24 is to add a production-oriented evaluation and guardrail layer around the existing Day 18 RAG pipeline. The implementation reuses the Day 18 modules for chunking, search, reranking, evaluation support, enrichment, and pipeline orchestration, then adds three independent evaluation/safety phases:

1. **Phase A — RAGAS evaluation on 50 questions**
2. **Phase B — LLM-as-Judge with swap-and-average and human calibration**
3. **Phase C — PII detection + input/output guardrails + adversarial and latency testing**

This separation allows model/retrieval quality and safety behavior to be tested independently while still evaluating the complete production stack.

---

## 2. Phase A — RAGAS Production Evaluation

The 50-question evaluation set contains:

- 20 factual questions;
- 20 multi-hop questions;
- 10 adversarial questions.

The real evaluation artifact reports backend `ragas`.

| Distribution | Faithfulness | Answer relevancy | Context precision | Context recall | Avg. score |
|---|---:|---:|---:|---:|---:|
| Factual | 0.975 | 0.841 | 0.967 | 1.000 | 0.946 |
| Multi-hop | 0.695 | 0.751 | 0.983 | 0.879 | 0.827 |
| Adversarial | 1.000 | 0.856 | 1.000 | 0.717 | 0.893 |
| **Overall weighted** | **0.868** | **0.808** | **0.980** | **0.895** | **0.888** |

### Main findings

- Overall faithfulness passes the 0.75 quality gate at **0.868**.
- Multi-hop faithfulness is the main quality weakness at **0.695**.
- Answer relevancy is the most common weakest metric (**27/50 questions**).
- Adversarial context recall is relatively weak at **0.717**, suggesting missing conflict/version evidence rather than excessive noisy retrieval.
- Seven of the bottom ten questions are multi-hop cases.

The detailed cluster diagnosis is documented in `analysis/failure_clusters.md`.

---

## 3. Phase B — LLM-as-Judge

The LLM judge uses the real `openai_llm_judge` backend with `gpt-4o-mini` through an OpenAI-compatible OpenRouter endpoint.

### Calibration results

- Human-labeled examples: **10**
- Agreement: **8/10 = 80%**
- Cohen's kappa: **0.615385**
- Interpretation: **Substantial agreement**

### Bias results

- Position-inconsistent cases: **2/10 = 20%**
- Decisive comparisons: **7**
- Longer-answer winner correlation: **7/7 = 100%**

The result supports using the judge for automated regression testing, but not as unquestioned ground truth. Swap-and-average remains necessary, and human recalibration should be retained for production changes.

Detailed analysis is in `analysis/bias_report.md`.

---

## 4. Phase C — Guardrail Stack

The guard stack combines PII detection and input/output safety controls around the Day 18 RAG pipeline.

### Adversarial evaluation

| Category | Passed | Total | Pass rate |
|---|---:|---:|---:|
| PII injection | 5 | 5 | 100% |
| Jailbreak | 5 | 5 | 100% |
| Off-topic | 5 | 5 | 100% |
| Prompt injection | 5 | 5 | 100% |
| **Total** | **20** | **20** | **100%** |

This exceeds both the assignment pass requirement (>=15/20) and the proposed CI target (>=18/20).

### Latency

| Layer | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---:|---:|---:|
| Presidio | 6.665 | 50.287 | 50.287 |
| NeMo | 0.028 | 796.888 | 796.888 |
| **Total guard** | **6.692** | **805.653** | **805.653** |

The guard stack is correct on the adversarial suite but **does not meet the <500 ms P95 latency target**. The NeMo evaluation path is the main measured contributor to the P95 tail and should be optimized before treating the latency gate as production-ready.

---

## 5. CI/CD Decision

| Gate | Threshold | Result | Status |
|---|---:|---:|---|
| Overall faithfulness | >=0.75 | 0.868 | PASS |
| Judge kappa | >=0.60 | 0.615 | PASS |
| Adversarial pass rate | >=90% | 100% | PASS |
| Guard P95 latency | <500 ms | 805.653 ms | **FAIL** |
| Unit tests | all pass | 40 passed | PASS |

The experiment is valid for the lab and clearly identifies a real performance limitation instead of masking it. For production deployment, guard latency is the first optimization priority.

---

## 6. Recommended Next Iteration

1. **Multi-hop retrieval/generation:** add query decomposition and an evidence-led answer synthesis step.
2. **Version-aware retrieval:** boost current/authoritative policies and suppress superseded versions in contradiction-sensitive queries.
3. **Answer directness:** enforce a compact answer contract that preserves requested entities, conditions, units and calculations.
4. **Judge calibration:** retain swap-and-average, expand human calibration data, and reduce verbosity sensitivity.
5. **Guard latency:** profile the NeMo path, apply cheap deterministic prefilters first, cache repeat decisions, and avoid unnecessary remote model calls.

---

## 7. Submission Artifacts

Required implementation and evidence in the repository:

```text
src/m1_chunking.py
src/m2_search.py
src/m3_rerank.py
src/m4_eval.py
src/m5_enrichment.py
src/pipeline.py
src/phase_a_ragas.py
src/phase_b_judge.py
src/phase_c_guard.py

reports/ragas_50q.json
reports/judge_results.json
reports/guard_results.json
reports/blueprint.md

analysis/failure_clusters.md
analysis/bias_report.md
```

`answers_50q.json` is generated locally by `setup_answers.py`. The starter repository ignores it by default; it should remain available locally for `check_lab.py`, and should only be force-added or submitted separately if the mentor explicitly requires it in Git.

---

## 8. Final Conclusion

The Lab 24 implementation successfully adds an end-to-end evaluation and safety stack to the Day 18 RAG system. RAG quality is strong overall, the LLM judge reaches substantial agreement with human labels, and all 20 adversarial guardrail cases pass. The evaluation also exposes actionable weaknesses instead of reporting only positive results: multi-hop faithfulness is below target, judge bias remains measurable, and guard P95 latency exceeds the production budget. These findings provide clear priorities for the next production iteration.
