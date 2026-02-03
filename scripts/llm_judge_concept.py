"""
SEAL Failure Analysis - LLM-as-a-Judge (HYBRID COMPARISON)
==========================================================
Logic:
1. Identify WRONG Passage (Rank 1).
2. Identify CORRECT Passage:
   - If SEAL retrieved the correct doc at Rank 2-10 (NP case), use THAT.
   - If SEAL missed it entirely (NN case), use the Dataset Ground Truth.
3. Ask LLM: "Why did Rank 1 beat the Correct Passage?"
"""

import json
import csv
import random
import time
import os
import re
import cohere

# ================= CONFIGURATION =================
OUTPUT_VERIFICATION = "generated_data/results_verification_hybrid.csv"
OUTPUT_CLASSIFICATION = "generated_data/results_failure_classification_hybrid.csv"

API_KEY = "vkNAU8NXHr5ozh9hq2v92JQWOemBpE4iwwAmFr8E" 
MODEL_NAME = "command-a-03-2025"

SAMPLE_SIZE_VERIFICATION = 30
SAMPLE_SIZE_FAILURES = 30
# =================================================

co = cohere.ClientV2(API_KEY)

def strip_pseudoqueries(text: str, datapath: str) -> str:
    """Strip pseudoquery markers from text if using Minder data."""
    if "minder_output.json" in datapath:
        # Remove || ... @@ patterns
        text = re.sub(r'\|\|[^@]*@@', '', text)
    return text


def strip_ngram_markers(ngram: str, datapath: str) -> str:
    """Strip pseudoquery markers from ngrams if using Minder data."""
    if "minder_output.json" in datapath:
        ngram = ngram.replace(" ||", "").strip()
    return ngram


def parse_keys_and_get_top(keys_field, top_n=10, datapath=""):
    try:
        if isinstance(keys_field, str):
            keys_list = json.loads(keys_field)
        else:
            keys_list = keys_field
        valid_keys = [k for k in keys_list if isinstance(k, list) and len(k) >= 3]
        valid_keys.sort(key=lambda x: x[2], reverse=True)
        top_keys = valid_keys[:top_n]
        formatted = []
        for k in top_keys:
            ngram_clean = strip_ngram_markers(k[0], datapath)
            formatted.append(f'"{ngram_clean}" (Score: {k[2]:.1f}, Freq: {k[1]})')
        return "\n".join(formatted)
    except:
        return "Error parsing keys"

def query_cohere(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    retries = 0
    while retries < 3:
        try:
            response = co.chat(
                model=MODEL_NAME,
                messages=messages
            )
            return response.message.content[0].text.strip()
        except Exception as e:
            if "429" in str(e).lower():
                print(f"  [!] Rate limit. Sleeping 65s...")
                time.sleep(65)
                retries += 1
            elif "404" in str(e):
                return "ERROR: Model not found"
            else:
                return f"ERROR: {str(e)}"
    return "ERROR: Max retries exceeded"

# --- PROMPTS ---

def get_ver_prompts(q, ans, text):
    sys = "You are an impartial judge evaluating an Information Retrieval system."
    user = f"""
    QUESTION: "{q}"
    EXPECTED ANSWER (from dataset): "{ans}"
    RETRIEVED PASSAGE: "{text}"
    
    TASK: Does the RETRIEVED PASSAGE actually contain the answer to the question?
    - Ignore exact string matching. Look for semantic meaning.
    - If the text mentions the topic but doesn't answer the specific question, say NO.
    
    Respond strictly:
    VERDICT: [YES/NO]
    REASON: [1 sentence explanation]
    """
    return sys, user

def get_class_prompts(q, ans, wrong_text, correct_text, top_keys, scenario_type):
    sys = "You are a forensic analyst for Search Engines. Diagnose why the wrong document was ranked #1."
    
    context_note = ""
    if scenario_type == "RECOVERABLE":
        context_note = "NOTE: The engine DID find the correct document, but ranked it lower than the wrong one."
    else:
        context_note = "NOTE: The engine completely failed to find the correct document in the top results."

    user = f"""
    QUERY: "{q}"
    CORRECT ANSWER: "{ans}"
    
    {context_note}
    
    [A] WRONG PASSAGE (Ranked #1):
    "{wrong_text[:500]}..."
    
    [B] CORRECT PASSAGE (Ground Truth / Lower Rank):
    "{correct_text[:500]}..."
    
    [KEYS USED FOR RANK #1] (Top Scoring N-Grams):
    {top_keys}
    
    TASK: Why did the model retrieve/rank [A] #1 instead of [B]?
    
    Classify into ONE category:
    
    1. NONSPECIFIC: The keys (e.g. 'History', 'Song') are too generic. They match [A] just as well as [B], so the model couldn't distinguish them.
    
    2. MISMATCH: The keys actively target [A] (e.g. Query asked for Director, keys focused on Cast). Or the keys align with [A] but are missing from [B].
    
    3. SCORING: The keys actually look relevant to [B] too, but [A] won anyway. (Likely a weighting issue or corpus frequency issue).
    
    Respond strictly:
    TYPE: [NONSPECIFIC / MISMATCH / SCORING]
    REASON: [1 sentence explanation]
    """
    return sys, user

def get_ground_truth_ids(entry):
    ids = set()
    if 'positive_ctxs' in entry:
        for ctx in entry['positive_ctxs']:
            ids.add(str(ctx.get('passage_id', '')).split('...')[0])
    return ids

def main(datapath="data/seal_output.json"):
    print(f"--- STARTING HYBRID ANALYSIS ---")

    if not os.path.exists(datapath):
        print("File not found.")
        return

    with open(datapath, 'r', encoding="utf-8") as f:
        data = json.load(f)
    
    artifact_cases = []
    failure_cases = []
    
    # 1. Prepare Data
    for entry in data:
        gold_ids = get_ground_truth_ids(entry)
        top_k = entry.get('ctxs', [])[:10]
        
        # Check if GT is in Top 10
        gt_found_idx = -1
        for idx, ctx in enumerate(top_k):
            if str(ctx.get('passage_id')) in gold_ids:
                gt_found_idx = idx
                break
        
        gt_in_top_k = (gt_found_idx != -1)
        
        # Check answer string
        ans_in_top_k = False
        answers = entry.get('answers', [])
        found_text = ""
        for c in top_k:
            txt = c.get('text', '') + ' ' + c.get('title', '')
            txt = strip_pseudoqueries(txt, datapath)
            if any(a.lower().strip() in txt.lower() for a in answers):
                ans_in_top_k = True
                found_text = txt
                break
        
        # EXP 1 Pool: Answer found, but NOT GT ID
        if not gt_in_top_k and ans_in_top_k:
            entry['eval_text'] = found_text
            artifact_cases.append(entry)
            
        # EXP 2 Pool: Rank 1 is NOT GT ID
        if top_k and (gt_found_idx != 0): # Rank 1 (index 0) is not GT
            entry['gt_found_index'] = gt_found_idx # Save where we found it (or -1)
            failure_cases.append(entry)

    # 2. Sample
    random.seed(42)
    verify_sample = random.sample(artifact_cases, min(SAMPLE_SIZE_VERIFICATION, len(artifact_cases)))
    class_sample = random.sample(failure_cases, min(SAMPLE_SIZE_FAILURES, len(failure_cases)))
    
    print(f"Sampled {len(verify_sample)} for Verification.")
    print(f"Sampled {len(class_sample)} for Classification.")

    # ---------------------------------------------------------
    # EXP 1: VERIFICATION
    # ---------------------------------------------------------
    print("\n=== RUNNING EXP 1: VERIFICATION ===")
    with open(OUTPUT_VERIFICATION, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'answers', 'verdict', 'raw_response'])
        writer.writeheader()
        
        for i, case in enumerate(verify_sample):
            print(f"Verifying {i+1}...")
            sys, user = get_ver_prompts(case['question'], case['answers'], case['eval_text'])
            resp = query_cohere(sys, user)
            
            verdict = "UNKNOWN"
            if "VERDICT: YES" in resp.upper(): verdict = "YES"
            elif "VERDICT: NO" in resp.upper(): verdict = "NO"
            
            writer.writerow({'question': case['question'], 'answers': case['answers'], 'verdict': verdict, 'raw_response': resp})
            f.flush()
            time.sleep(4)

    # ---------------------------------------------------------
    # EXP 2: CLASSIFICATION
    # ---------------------------------------------------------
    print("\n=== RUNNING EXP 2: CLASSIFICATION ===")
    with open(OUTPUT_CLASSIFICATION, 'w', newline='', encoding='utf-8') as f:
        # Added 'scenario' to output so you know if it was NP or NN
        writer = csv.DictWriter(f, fieldnames=['question', 'scenario', 'failure_type', 'raw_response'])
        writer.writeheader()
        
        for i, case in enumerate(class_sample):
            print(f"Classifying {i+1}...")
            
            # [A] Wrong Passage (Rank 1)
            wrong_doc = case['ctxs'][0]
            wrong_text = wrong_doc.get('title', '') + ": " + strip_pseudoqueries(wrong_doc.get('text', ''), datapath)

            # [B] Correct Passage
            gt_idx = case.get('gt_found_index', -1)

            scenario = "UNRECOVERABLE (NN)"
            correct_text = ""

            if gt_idx > 0:
                # RECOVERABLE (NP): The correct doc IS in the list (e.g. Rank 2)
                scenario = f"RECOVERABLE (Rank {gt_idx+1})"
                correct_doc = case['ctxs'][gt_idx]
                correct_text = correct_doc.get('title', '') + ": " + strip_pseudoqueries(correct_doc.get('text', ''), datapath)
            else:
                # UNRECOVERABLE (NN): Correct doc NOT in list. Use Dataset GT.
                if case.get('positive_ctxs'):
                    p = case['positive_ctxs'][0]
                    correct_text = p.get('title', '') + ": " + strip_pseudoqueries(p.get('text', ''), datapath)
                else:
                    correct_text = "No Ground Truth Text Available."

            # Get Top Keys from Rank 1 (Why did this win?)
            top_keys_str = parse_keys_and_get_top(wrong_doc.get('keys', []), top_n=8, datapath=datapath)
            
            sys, user = get_class_prompts(
                case['question'],
                case['answers'],
                wrong_text,
                correct_text,
                top_keys_str,
                "RECOVERABLE" if gt_idx > 0 else "UNRECOVERABLE"
            )
            
            resp = query_cohere(sys, user)
            
            ftype = "OTHER"
            if "TYPE:" in resp:
                try: ftype = resp.split("TYPE:")[1].split()[0].strip()
                except: pass
            
            writer.writerow({
                'question': case['question'], 
                'scenario': scenario,
                'failure_type': ftype, 
                'raw_response': resp
            })
            f.flush()
            time.sleep(4)

    print("\nDone.")

if __name__ == "__main__":
    main()