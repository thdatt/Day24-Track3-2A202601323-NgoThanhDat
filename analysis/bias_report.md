# LLM Judge Bias Report — Phase B

**Backend:** `openai_llm_judge`
**Judge model:** `gpt-4o-mini`
**Cohen's kappa:** **0.615** (substantial)
**Human/judge agreement:** 8/10

## Bias metrics

- Position bias rate: **20.0%** (2/10)
- Verbosity bias rate: **100.0%**
- A wins while A is longer: 0
- B wins while B is longer: 7
- Decisive comparisons: 7

**Interpretation:** Position bias is low on this evaluation sample.

> If the backend is `offline_reference_proxy`, this is a smoke-test result, not final LLM-judge evidence.
