# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Student:** Ngo Thanh Dat — 2A202601323  
**Project:** Day24-Track3-2A202601323-NgoThanhDat

> This file is generated from this repository's own Phase A/B/C reports. Sample metrics from other repositories are not used.

## 1. Guard Stack Pipeline

| Layer | Tool | Target P95 | Failure Action |
|---|---|---:|---|
| PII Detection | Presidio + VN recognizers | < 10 ms | Reject + anonymize + log |
| Topic/Jailbreak | NeMo Input Rail | < 300 ms | Block + reason |
| RAG Pipeline | Day 18 Production RAG | < 2000 ms | Grounded fallback / error log |
| Output Check | NeMo Output Rail + PII scan | < 300 ms | Redact/block + log |

## 2. CI Gates

- [x] RAGAS faithfulness >= 0.75 on 50q using the real RAGAS backend
- [x] Adversarial pass rate >= 90% (18/20 target)
- [ ] P95 total guard latency < 500 ms
- [ ] `pytest tests/ -v` passes (verify in CI/local terminal)
- [ ] `python check_lab.py` passes (verify after all artifacts are generated)

## 3. Monitoring

- P95 guard latency: **805.653 ms**
- Adversarial pass rate: **100.0%**
- Worst RAGAS metric: **answer_relevancy**
- Dominant failure distribution: **factual**
- RAGAS backend: **ragas**
- Judge backend/model: **openai_llm_judge / gpt-4o-mini**

## 4. Production Actions

1. Run the 50-question RAGAS suite on every retrieval/generation change and block material regressions.
2. Keep swap-and-average for release evaluation to detect position bias.
3. Run the 20-case adversarial suite on every guardrail/configuration change.
4. Persist per-layer latency, guardrail block reasons, RAGAS distribution metrics, and judge agreement for trend monitoring.
5. Treat `offline_proxy`, `offline_reference_proxy`, and deterministic fallbacks as development smoke tests, never as production evidence.
