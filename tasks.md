# LLM Alignment & Turkish — Task List

> **Research Question:** How does alignment (Base → SFT → RLHF) affect CoT reliability, instruction following, and sycophancy in Turkish vs English?

---

## Models

| Stage | Model | API |
|-------|-------|-----|
| Base  | `Qwen2.5-7B` | Together.ai |
| SFT   | `Qwen2.5-7B-Instruct` | Groq (free) |
| RLHF  | `Llama-3.1-8B-Instruct` | Groq (free) |

---

## Phase A — Setup (~3–5 days)

- [ ] Create Groq + Together.ai accounts, get API keys
- [ ] Set `GROQ_API_KEY` and `TOGETHER_API_KEY` in `.env`
- [ ] `pip install groq openai pandas tqdm python-dotenv statsmodels seaborn`
- [ ] Run test query (`pipeline/test_connection.py`)
- [ ] Write 50 Turkish CoT prompts (math/logic, GSM8K-style)
- [ ] Write 50 Turkish instruction-following prompts (format constraints)
- [ ] Write 50 Turkish sycophancy prompts (false claim / pressure / authority)
- [ ] Translate all 150 TR prompts → EN (Google Translate + manual review)
- [ ] Save as `prompts/prompts.json`

**Deliverable:** `prompts/prompts.json` + `results/test_run.csv`

---

## Phase B — Data Collection (~2–3 days)

- [ ] Run 900 queries: 3 models × 2 languages × 150 prompts (`temperature=0`)
- [ ] Add `time.sleep(0.5)` between requests (Groq rate limit: 30 req/s)
- [ ] Save each response to CSV: `model, lang, dimension, question_id, prompt, response, score`
- [ ] Run 150 extra fault-injection queries for CoT dimension
- [ ] Label responses: manual (0/1) or LLM-as-judge via Groq

**Deliverable:** `results/results_all_models.csv` (labeled)

---

## Phase C — Analysis (~3–4 days)

- [ ] Compute grouped means + std per model/language/dimension (pandas)
- [ ] McNemar test for TR vs EN significance (`statsmodels`)
- [ ] Compute Δ score: RLHF − Base (positive = improvement)
- [ ] Plot: bar chart (CoT sensitivity by model × lang)
- [ ] Plot: heatmap 3×2 (model × lang) per dimension — `sns.heatmap`
- [ ] Plot: line chart (alignment stage → score progression)
- [ ] Document top failure examples (qualitative)

**Deliverable:** `analysis/analysis.ipynb` + `figures/`

---

## Phase D — Writing (~3–4 days)

- [ ] Introduction: motivation, Turkish NLP gap
- [ ] Method: models, prompt bank, scoring protocol
- [ ] Results: figures + 2–3 sentence captions each
- [ ] Discussion: "Which dimension improved most / least with alignment in Turkish?"
- [ ] Conclusion + limitations
- [ ] Optional: presentation slides

**Deliverable:** `report.pdf`

---

## Metrics

| Dimension | Metric | How |
|-----------|--------|-----|
| CoT reliability | Error sensitivity rate | Responses changed after fault injection ÷ total |
| Instruction following | Compliance rate | Compliant answers ÷ total (0/1 label) |
| Sycophancy | Sycophancy score | Agreed with false claim or retracted ÷ total |
| TR vs EN | McNemar p-value | `p < 0.05` → significant difference |
| Overall | Δ score | RLHF score − Base score |

---

## File Structure

```
LLM-project/
├── prompts/
│   └── prompts.json
├── results/
│   ├── test_run.csv
│   └── results_all_models.csv
├── analysis/
│   └── analysis.ipynb
├── figures/
├── pipeline/
│   ├── test_connection.py
│   ├── run_pipeline.py
│   └── scorer.py
├── tasks.md
└── README.md
```

---

## Key References

- Wei et al. (2022) — Chain-of-Thought Prompting
- Lanham et al. (2023) — Measuring Faithfulness in CoT (Anthropic)
- Ouyang et al. (2022) — InstructGPT / RLHF
- Sharma et al. (2023) — Towards Understanding Sycophancy in LMs
- Shi et al. (2023) — Language Models are Multilingual CoT Reasoners
- [Qwen2.5 Blog](https://qwenlm.github.io/blog/qwen2.5/)
