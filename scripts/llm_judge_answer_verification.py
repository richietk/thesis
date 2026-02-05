"""
LLM-as-Judge: Verify "Answer in Different Passage" Cases
=========================================================
For the 426 cases where the answer string appears in a top-2 passage
that is NOT the annotated ground truth, determine whether the passage
genuinely answers the question or is a coincidental string match.

Uses Claude Sonnet via Anthropic API with structured outputs.
"""

import anthropic
import csv
import json
import os
import time
import ijson
from scripts.utils import strip_pseudoqueries, get_dataset_name

# ================= CONFIGURATION =================
# Note: INPUT_CSV and OUTPUT_CSV are now set dynamically in main() based on datapath

API_KEY = os.environ["NOTHINGITDOENSTEXIST"]
#ANTHROPIC_API_KEY
MODEL = "claude-sonnet-4-5-2025-0929"

RATE_LIMIT_DELAY = 0.5  # seconds between requests
# =================================================

client = anthropic.Anthropic(api_key=API_KEY)

# Structured output schema
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["YES", "NO", "PARTIAL"]
        },
        "reason": {
            "type": "string"
        }
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False
}


def get_verification_prompt(question: str, answer: str, passage_title: str, passage_text: str) -> str:
    """Generate the classification prompt."""
    return f"""Question: "{question}"
Expected answer: "{answer}"
Retrieved passage title: "{passage_title}"
Retrieved passage text: "{passage_text}"

Does this passage answer the question?
- YES: The passage directly answers the question.
- NO: The answer string appears coincidentally but does not answer the question.
- PARTIAL: Contains relevant information but requires inference.

The answer explanation should not exceed 100 words.
"""


def query_claude(prompt: str, token_limit: int = 150) -> dict:
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=token_limit, # Start at 150
            betas=["structured-outputs-2025-11-13"],
            messages=[{"role": "user", "content": prompt}],
            output_format={"type": "json_schema", "schema": OUTPUT_SCHEMA}
        )
        return json.loads(response.content[0].text)
    
    except (json.JSONDecodeError, Exception) as e:
        # If it was a JSON error and we haven't hit the ceiling yet, retry
        if token_limit < 600:
            print(f"  Truncated response at {token_limit} tokens. Retrying with 600...")
            return query_claude(prompt, token_limit=600)
        return {"verdict": "ERROR", "reason": f"API/Parse Error: {str(e)}"}
    except anthropic.RateLimitError:
        print("  Rate limited, waiting 60s...")
        time.sleep(60)
        return query_claude(prompt)
    except Exception as e:
        return {"verdict": "ERROR", "reason": str(e)}



def load_json_data_stream(json_path: str):
    """Stream JSON and index by question on-the-fly for lookup."""
    json_index = {}
    with open(json_path, 'r', encoding='utf-8') as f:
        for entry in ijson.items(f, "item"):
            question = entry.get("question")
            if question:
                json_index[question] = entry
    return json_index


# Change the function to check the whole list
def find_answer_passage(entry: dict, answers: list, top_k: int = 2, datapath: str = "") -> tuple:
    """Find the first passage containing ANY of the answer strings."""
    for ctx in entry.get('ctxs', [])[:top_k]:
        # Concatenate Title and Text
        passage_text = ctx.get('text', '') + ' ' + ctx.get('title', '')
        # Strip pseudoqueries if Minder data
        passage_text = strip_pseudoqueries(passage_text, datapath)
        # Normalize exactly like the analysis script
        normalized_passage = passage_text.lower().strip()

        for ans in answers:
            # Normalize the answer string
            if ans.lower().strip() in normalized_passage:
                return ctx.get('title', ''), ctx.get('text', '')
    return None, None


def main(datapath="data/seal_output.json"):
    import sys

    script_name = "llm_judge_answer_verification"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        INPUT_CSV = f"{output_dir}/answer_location_analysis_{dataset_name}.csv"
        OUTPUT_CSV = f"{output_dir}/llm_judge_results_{dataset_name}.csv"

        if not os.path.exists(INPUT_CSV):
            print(f"error: running {script_name} {INPUT_CSV} not found")
            return

        # Load the full JSON for passage text
        json_data = load_json_data_stream(datapath)

        # Load CSV and filter for "answer_in_different_passage" cases
        cases_to_verify = []
        with open(INPUT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['outcome'] == 'answer_in_different_passage':
                    cases_to_verify.append(row)

        # Process and write results
        results = {'YES': 0, 'NO': 0, 'PARTIAL': 0, 'ERROR': 0}

        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'question', 'answer', 'passage_title', 'passage_text', 'verdict', 'reason'
            ])
            writer.writeheader()

            for i, case in enumerate(cases_to_verify):
                question = case['question']
                # Load the full list of answers
                try:
                    answers = eval(case['answers'])
                except:
                    answers = [case['answers']] # Fallback if eval fails

                # Get entry from JSON
                entry = json_data.get(question)
                if not entry:
                    continue

                # SEARCH USING THE FULL LIST
                title, text = find_answer_passage(entry, answers, top_k=2, datapath=datapath)

                if not text:
                    continue

                # If found, prepare the first answer for the prompt (Claude only needs one)
                primary_answer = answers[0] if answers else ""
                prompt = get_verification_prompt(question, primary_answer, title, text)
                response = query_claude(prompt)

                verdict = response.get("verdict", "ERROR")
                reason = response.get("reason", "")

                results[verdict] = results.get(verdict, 0) + 1

                writer.writerow({
                    'question': question,
                    'answer': primary_answer,
                    'passage_title': title,
                    'passage_text': text,
                    'verdict': verdict,
                    'reason': reason
                })
                f.flush()

                time.sleep(RATE_LIMIT_DELAY)

        # Calculate summary statistics
        total = sum(results.values())
        genuine = results['YES']
        partial = results['PARTIAL']
        base_success = 3416  # from top-2 analysis
        total_queries = 6515

        adjusted_strict = (base_success + genuine) / total_queries * 100
        adjusted_lenient = (base_success + genuine + partial) / total_queries * 100

        # Build JSON output
        output_data = {
            "cases_verified": len(cases_to_verify),
            "total_processed": total,
            "verdict_counts": {
                "genuine_yes": results['YES'],
                "coincidental_no": results['NO'],
                "borderline_partial": results['PARTIAL'],
                "errors": results['ERROR']
            },
            "verdict_percentages": {
                "genuine_yes_pct": float(results['YES'] / total * 100) if total > 0 else 0.0,
                "coincidental_no_pct": float(results['NO'] / total * 100) if total > 0 else 0.0,
                "borderline_partial_pct": float(results['PARTIAL'] / total * 100) if total > 0 else 0.0
            },
            "recall_adjustment": {
                "base_top2_recall_pct": 52.4,
                "adjusted_strict_yes_only_pct": float(adjusted_strict),
                "adjusted_lenient_yes_and_partial_pct": float(adjusted_lenient)
            }
        }

        # Save JSON output
        output_json = os.path.join(output_dir, f"{script_name}_results.json")
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"success running {script_name}")

    except Exception as e:
        print(f"error: running {script_name} {e}")
        raise


if __name__ == "__main__":
    main()
