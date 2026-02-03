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

# ================= CONFIGURATION =================
INPUT_CSV = "generated_data/answer_location_analysis.csv"
OUTPUT_CSV = "generated_data/llm_judge_results.csv"

API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-5-20250929"

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
    print(f"Streaming {json_path}...")
    json_index = {}
    with open(json_path, 'r', encoding='utf-8') as f:
        for entry in ijson.items(f, "item"):
            question = entry.get("question")
            if question:
                json_index[question] = entry
    return json_index


# Change the function to check the whole list
def find_answer_passage(entry: dict, answers: list, top_k: int = 2) -> tuple:
    """Find the first passage containing ANY of the answer strings."""
    for ctx in entry.get('ctxs', [])[:top_k]:
        # Concatenate Title and Text
        passage_text = ctx.get('text', '') + ' ' + ctx.get('title', '')
        # Normalize exactly like the analysis script
        normalized_passage = passage_text.lower().strip()
        
        for ans in answers:
            # Normalize the answer string
            if ans.lower().strip() in normalized_passage:
                return ctx.get('title', ''), ctx.get('text', '')
    return None, None


def main(datapath="data/seal_output.json"):
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Run answer_location_analysis.py first.")
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

    print(f"Found {len(cases_to_verify)} cases to verify")

    # Process and write results
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

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
                print(f"  [{i+1}] Question not found in JSON index, skipping")
                continue

            # SEARCH USING THE FULL LIST
            title, text = find_answer_passage(entry, answers, top_k=2)
            
            if not text:
                # If it still fails, it's likely a Duplicate Question overwrite issue
                print(f"  [{i+1}] Warning: Still could not find answer for: {question[:30]}")
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

            print(f"  [{i+1}/{len(cases_to_verify)}] {verdict}: {question[:50]}...")
            time.sleep(RATE_LIMIT_DELAY)

    # Print summary
    total = sum(results.values())
    print("\n" + "=" * 60)
    print(f"LLM-AS-JUDGE RESULTS (N={total})")
    print("=" * 60)
    print(f"  Genuine retrieval (YES):      {results['YES']:>4} ({results['YES']/total*100:.1f}%)")
    print(f"  Coincidental match (NO):      {results['NO']:>4} ({results['NO']/total*100:.1f}%)")
    print(f"  Borderline (PARTIAL):         {results['PARTIAL']:>4} ({results['PARTIAL']/total*100:.1f}%)")
    print(f"  Errors:                       {results['ERROR']:>4}")
    print("=" * 60)

    # Calculate adjusted recall
    genuine = results['YES']
    partial = results['PARTIAL']
    base_success = 3416  # from top-2 analysis
    total_queries = 6515

    adjusted_strict = (base_success + genuine) / total_queries * 100
    adjusted_lenient = (base_success + genuine + partial) / total_queries * 100

    print(f"\nRecall Adjustment:")
    print(f"  Base top-2 recall:                    52.4%")
    print(f"  Adjusted (YES only):                  {adjusted_strict:.1f}%")
    print(f"  Adjusted (YES + PARTIAL):             {adjusted_lenient:.1f}%")

    print(f"\nResults saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
