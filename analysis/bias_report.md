# LLM Judge Bias Report — Phase B

**Student:** Ngo Thanh Dat — 2A202601323  
**Backend:** `openai_llm_judge`  
**Judge model:** `gpt-4o-mini`  
**OpenAI-compatible endpoint:** `https://openrouter.ai/api/v1`

---

## 1. Calibration Against Human Labels

The judge was calibrated on 10 labeled examples from `human_labels_10q.json`.

| Question ID | Human label | Judge label | Agreement |
|---:|---:|---:|---|
| 1 | 1 | 0 | No |
| 5 | 0 | 0 | Yes |
| 12 | 1 | 0 | No |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 1 | Yes |
| 29 | 0 | 0 | Yes |
| 33 | 1 | 1 | Yes |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 1 | Yes |
| 50 | 0 | 0 | Yes |

**Observed agreement:** **8/10 = 80%**  
**Cohen's kappa:** **0.615385**  
**Interpretation:** **Substantial agreement**

The judge clears a reasonable >=0.60 calibration threshold, but the two disagreements show that automated judging should remain a regression signal rather than an unquestioned replacement for human review.

---

## 2. Calibration Disagreements

### Question 1 — Marriage leave

- Human label: `1`
- Judge label: `0`
- Judge concern: the candidate stated the correct three paid working days but omitted that the leave does not reduce annual leave.

**Interpretation:** the judge applies a stricter completeness criterion than the human label for a question that primarily asks for the number of days. This indicates a possible **completeness-over-directness bias**.

### Question 12 — Tet bonus

- Human label: `1`
- Judge label: `0`
- Judge concern: the candidate gave the correct one-month minimum but omitted the six-month eligibility condition/pro-rata detail.

**Interpretation:** again, the judge penalizes an answer that contains the requested headline value but omits eligibility context. The evaluation prompt should define when omitted conditions are critical versus optional.

---

## 3. Swap-and-Average / Position Bias

Ten pairwise comparisons were evaluated in both answer orders.

- Position-inconsistent comparisons: **2 / 10**
- Position bias rate: **20.0%**
- Position-consistent comparisons: **8 / 10**

The two inconsistent cases were:

1. Marriage leave: one pass preferred the more complete answer while the swapped pass resulted in a tie.
2. Senior employee leave + salary: one pass preferred the more detailed answer while the swapped pass resulted in a tie.

**Conclusion:** swap-and-average is necessary. A 20% inconsistency rate is too large to rely on a single ordering for release-quality evaluation.

---

## 4. Verbosity Bias

The generated bias report contains:

- Decisive comparisons: **7**
- A wins while A is longer: **0**
- B wins while B is longer: **7**
- Reported verbosity-bias rate: **100.0%**

This result must be interpreted carefully. It shows a perfect correlation between the winning answer and the longer `B` answer within the seven decisive comparisons, but **correlation is not proof that length caused the judge decision**. In several pairs, `B` is also more factually complete.

Still, this is a meaningful warning because the judge repeatedly rewards additional detail. The prompt should explicitly distinguish:

- critical completeness;
- useful but optional context;
- unnecessary verbosity.

A production judge should not reward a longer answer when a shorter answer fully satisfies the user request.

---

## 5. Judge Reliability Summary

| Dimension | Result | Assessment |
|---|---:|---|
| Human agreement | 80% | Good but not perfect |
| Cohen's kappa | 0.615 | Substantial |
| Position inconsistency | 20% | Needs monitoring/improvement |
| Verbosity correlation | 100% of decisive pairs | Strong warning |
| Swap-and-average | Enabled | Required for release evaluation |

---

## 6. Recommended Improvements

1. **Keep swap-and-average for release gates.** Single-order evaluation is not reliable enough given the observed 20% inconsistency.
2. **Clarify the scoring rubric.** Explicitly state when a condition is essential to correctness and when concise answers should receive full credit.
3. **Normalize answer length where practical.** Compare answers with similar information budgets during judge calibration to separate completeness from verbosity effects.
4. **Expand human calibration labels.** Ten examples are enough for the lab but small for production. Add cases covering short-but-complete answers, version conflicts, negations, and calculations.
5. **Track kappa and bias over time.** Judge-model or prompt changes should trigger recalibration rather than assuming historical reliability transfers automatically.

---

## 7. Conclusion

The LLM judge is **usable but not bias-free**. Cohen's kappa of **0.615** supports using it for automated regression testing, while the **20% position inconsistency** and **100% verbosity correlation among decisive comparisons** justify maintaining swap-and-average and periodic human calibration. This is a stronger production posture than treating judge output as ground truth.
