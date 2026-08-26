# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Student:** Ngo Thanh Dat — 2A202601323  
**Project:** Day24-Track3-2A202601323-NgoThanhDat  
**Evaluation date:** 2026-08-26

> All metrics in this document come from this repository's generated Phase A/B/C artifacts. No metrics from the reference/sample repository are reused.

---

## 1. Production Guard Stack Architecture

```text
User Input
    |
    v
[Presidio PII Scan]
    |  Detect: VN_CCCD / VN_PHONE / EMAIL and other configured PII
    |  Failure action: reject or anonymize + log
    v
[NeMo Input Rail]
    |  Detect: jailbreak / prompt injection / off-topic / sensitive request
    |  Failure action: block + safe reason
    v
[Day 18 Production RAG]
    |  M1 Chunking -> M2 Search -> M3 Rerank -> Generation
    |  Failure action: grounded fallback / error logging
    v
[NeMo Output Rail + PII Check]
    |  Detect unsafe or sensitive output before returning it
    |  Failure action: redact or block + log
    v
User Response
```

The Day 24 stack intentionally wraps the existing Day 18 production RAG instead of rewriting it. Evaluation and safety are separate layers so retrieval/generation changes can be regression-tested without coupling them to guardrail implementation.

---

## 2. Measured Evaluation Summary

### Phase A — RAGAS 50-question evaluation

| Distribution | Count | Faithfulness | Answer relevancy | Context precision | Context recall | Avg. score |
|---|---:|---:|---:|---:|---:|---:|
| Factual | 20 | 0.975 | 0.841 | 0.967 | 1.000 | 0.946 |
| Multi-hop | 20 | 0.695 | 0.751 | 0.983 | 0.879 | 0.827 |
| Adversarial | 10 | 1.000 | 0.856 | 1.000 | 0.717 | 0.893 |
| **Weighted overall** | **50** | **0.868** | **0.808** | **0.980** | **0.895** | **0.888** |

**Backend:** `ragas`  
**Worst metric by failure-count matrix:** `answer_relevancy`  
**Generated dominant failure distribution:** `factual`

Important nuance: although the failure-count matrix marks `factual` as the dominant cluster, **multi-hop has the lowest average score (0.827) and the lowest faithfulness (0.695)**. Therefore production monitoring should use both absolute failure counts and normalized per-distribution metrics.

### Phase B — LLM-as-Judge

| Metric | Result |
|---|---:|
| Backend | `openai_llm_judge` |
| Judge model | `gpt-4o-mini` |
| API endpoint | OpenRouter-compatible OpenAI API |
| Human/judge agreement | 8 / 10 |
| Cohen's kappa | **0.615** |
| Kappa interpretation | **Substantial agreement** |
| Position inconsistency | 2 / 10 (**20.0%**) |
| Verbosity-bias correlation | **100.0%** of 7 decisive comparisons |

The judge is usable as an automated regression signal, but it is not bias-free. Swap-and-average detected two position-inconsistent cases, and every decisive comparison favored the longer answer. This does not prove that length caused the preference, but it is a strong enough correlation to require continued calibration.

### Phase C — Guardrail evaluation

| Metric | Result |
|---|---:|
| Adversarial cases | 20 |
| Passed | **20 / 20** |
| Pass rate | **100.0%** |
| PII injection | 5 / 5 passed |
| Jailbreak | 5 / 5 passed |
| Off-topic | 5 / 5 passed |
| Prompt injection | 5 / 5 passed |

Guard latency measured by the lab benchmark:

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Target |
|---|---:|---:|---:|---:|
| Presidio | 6.665 | 50.287 | 50.287 | < 100 ms operational target |
| NeMo input evaluation | 0.028 | 796.888 | 796.888 | < 300 ms |
| **Total guard** | **6.692** | **805.653** | **805.653** | **< 500 ms** |

**Latency budget status: FAIL.** Safety accuracy passed strongly, but P95 total guard latency exceeded the 500 ms target by approximately **305.653 ms**. The main measured contributor is the NeMo path.

---

## 3. CI Quality Gates

The following gates should run before merging a retrieval, generation, judge, or guardrail change to `main`.

- [x] Overall RAGAS faithfulness >= 0.75 on the 50-question evaluation set (**0.868**)
- [x] Adversarial suite pass rate >= 90% (**20/20 = 100%**)
- [ ] P95 total guard latency < 500 ms (**805.653 ms**, optimization required)
- [x] Unit test suite passes in the verified local run (**40 passed**)
- [ ] `python check_lab.py` must be run after local `answers_50q.json` exists; do not mark this gate passed from an artifact snapshot that omits the ignored file

Recommended additional production gates:

- [ ] Multi-hop faithfulness >= 0.75 (current: **0.695**)
- [x] Judge Cohen's kappa >= 0.60 (current: **0.615**)
- [ ] Position inconsistency <= 10% (current: **20%**)

---

## 4. Suggested GitHub Actions Gate

```yaml
name: RAG Quality and Guardrail Gates

on:
  pull_request:
  push:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Unit tests
        run: pytest tests/ -v

      - name: Phase A - RAGAS
        run: python src/phase_a_ragas.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Phase B - LLM Judge
        run: python src/phase_b_judge.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Phase C - Guardrails
        run: python src/phase_c_guard.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

The CI job should fail when a required quality/safety threshold regresses. The current latency result should remain visible as a failing performance gate rather than being hidden or replaced with a sample value.

---

## 5. Production Monitoring

| Metric | Recommended alert threshold | Current lab result | Action |
|---|---:|---:|---|
| Overall faithfulness | < 0.75 | 0.868 | Investigate grounding/retrieval regression |
| Multi-hop faithfulness | < 0.75 | **0.695** | Query decomposition + better grounded generation |
| Answer relevancy | < 0.75 | 0.808 overall | Tighten direct-answer generation prompt |
| Adversarial pass rate | < 90% | 100% | Review attack vectors and rail rules |
| Guard P95 latency | > 500 ms | **805.653 ms** | Profile NeMo path, cache or reduce remote checks |
| Judge kappa | < 0.60 | 0.615 | Human recalibration and prompt review |
| Position inconsistency | > 10% | **20%** | Continue swap-and-average; recalibrate judge prompt |
| PII detections | abnormal spike | 5/5 test cases blocked | Security review and source analysis |

---

## 6. Improvement Plan

### P0 — Reduce guardrail latency
The 100% safety pass rate is strong, but the P95 guard latency does not meet the lab CI target. Profile the NeMo path first because its measured P95 (796.888 ms) dominates total P95. Candidate improvements include deterministic prefilters before model-backed rails, caching repeated policy decisions, minimizing remote-model calls, and separating fast synchronous blocks from slower secondary analysis.

### P1 — Improve multi-hop faithfulness
Multi-hop faithfulness is 0.695, below the proposed 0.75 production threshold. Improve query decomposition, retrieve evidence for each sub-question, preserve source/version metadata, and force generation to cite only retrieved evidence.

### P1 — Improve answer relevancy
`answer_relevancy` is the most frequent weakest metric (27/50). Generation should preserve the exact intent, entity, condition, unit, and requested format while avoiding unnecessary policy commentary.

### P2 — Calibrate the LLM judge
Cohen's kappa is substantial (0.615), but 2/10 comparisons were position-inconsistent and decisive results correlate strongly with answer length. Maintain swap-and-average for release evaluation, strengthen conciseness criteria, and periodically recalibrate against fresh human labels.

---

## 7. Release Decision

**Quality:** PASS overall, with a multi-hop faithfulness warning.  
**Safety:** PASS (20/20 adversarial cases).  
**Judge calibration:** PASS at the >=0.60 kappa threshold, with bias monitoring required.  
**Latency:** FAIL against the <500 ms P95 target.

For a classroom submission, the experiment is complete and the latency miss is correctly documented as an observed production limitation. For a real production release, optimize the guard path before enforcing the latency gate as mandatory.
