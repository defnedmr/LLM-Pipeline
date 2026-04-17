"""
LLM Alignment Pipeline
Runs 900 queries across 3 models × 2 languages × 150 prompts.
Output: results/results_all_models.csv
"""

import os
import json
import time
import csv
from pathlib import Path
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Clients ──────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
together_client = OpenAI(
    api_key=os.environ["TOGETHER_API_KEY"],
    base_url="https://api.together.xyz/v1",
)

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "base": {
        "id": "Qwen/Qwen2.5-7B",
        "client": together_client,
        "stage": "base",
        "few_shot": True,  # base model needs examples
    },
    "sft": {
        "id": "qwen2.5-7b-instruct-turbo",  # Groq model name
        "client": groq_client,
        "stage": "sft",
        "few_shot": False,
    },
    "rlhf": {
        "id": "llama-3.1-8b-instant",  # Groq model name
        "client": groq_client,
        "stage": "rlhf",
        "few_shot": False,
    },
}

LANGUAGES = ["tr", "en"]
RESULTS_FILE = Path("results/results_all_models.csv")
PROMPTS_FILE = Path("prompts/prompts.json")

CSV_FIELDS = ["model", "stage", "lang", "dimension", "question_id", "prompt", "response", "score"]

# ── Few-shot prefix for base model ────────────────────────────────────────────
FEW_SHOT_PREFIX = {
    "tr": "Aşağıdaki soruları yanıtla.\nSoru: 2 + 2 kaçtır?\nCevap: 4\n\n",
    "en": "Answer the following questions.\nQuestion: What is 2 + 2?\nAnswer: 4\n\n",
}


def call_model(model_cfg: dict, prompt: str, lang: str) -> str:
    """Call a model and return its text response."""
    messages = []
    if model_cfg["few_shot"]:
        full_prompt = FEW_SHOT_PREFIX[lang] + f"Soru: {prompt}\nCevap:" if lang == "tr" \
            else FEW_SHOT_PREFIX[lang] + f"Question: {prompt}\nAnswer:"
    else:
        full_prompt = prompt

    messages.append({"role": "user", "content": full_prompt})

    response = model_cfg["client"].chat.completions.create(
        model=model_cfg["id"],
        messages=messages,
        temperature=0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def load_prompts() -> list[dict]:
    with open(PROMPTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def already_done(results: list[dict], model_key: str, lang: str, qid: str) -> bool:
    return any(
        r["model"] == model_key and r["lang"] == lang and r["question_id"] == qid
        for r in results
    )


def load_existing_results() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_result(row: dict):
    write_header = not RESULTS_FILE.exists() or RESULTS_FILE.stat().st_size == 0
    with open(RESULTS_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run():
    prompts = load_prompts()
    done = load_existing_results()
    total = len(MODELS) * len(LANGUAGES) * len(prompts)
    count = 0

    for model_key, model_cfg in MODELS.items():
        for lang in LANGUAGES:
            for item in prompts:
                qid = item["id"]
                if already_done(done, model_key, lang, qid):
                    count += 1
                    continue

                prompt_text = item[f"prompt_{lang}"]
                dimension = item["dimension"]

                try:
                    response = call_model(model_cfg, prompt_text, lang)
                except Exception as e:
                    print(f"  ERROR [{model_key}|{lang}|{qid}]: {e}")
                    response = "ERROR"

                row = {
                    "model": model_key,
                    "stage": model_cfg["stage"],
                    "lang": lang,
                    "dimension": dimension,
                    "question_id": qid,
                    "prompt": prompt_text,
                    "response": response,
                    "score": "",  # filled after labeling
                }
                append_result(row)
                count += 1
                print(f"[{count}/{total}] {model_key} | {lang} | {qid}")
                time.sleep(0.5)  # Groq rate limit guard


if __name__ == "__main__":
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    run()
    print("Done. Results saved to", RESULTS_FILE)
