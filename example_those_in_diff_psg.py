"""
Inspect 'Answer in Different Passage' Cases (Full Text)
=======================================================
1. Reads 'answer_location_analysis.csv' to identify queries categorized 
   as 'answer_in_different_passage'.
2. Loads the full data from 'data/seal_output.json'.
3. Prints the full comparison (Question, GT, Top 2 Retrieved) without text truncation.
"""

import json
import csv
import os

def load_target_questions(csv_path):
    """
    Reads the CSV and returns a set of questions where the outcome 
    was 'answer_in_different_passage'.
    """
    target_questions = set()
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return target_questions

    print(f"Loading targets from {csv_path}...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter for the specific outcome category
            if row.get('outcome') == 'answer_in_different_passage':
                target_questions.add(row.get('question', '').strip())
    
    print(f"Found {len(target_questions)} queries categorized as 'answer_in_different_passage'.")
    return target_questions

def display_comparison(entry, index, total):
    """Prints the comparison for a single entry with FULL TEXT."""
    print("=" * 80)
    print(f"CASE {index}/{total}")
    print(f"QUESTION: {entry.get('question')}")
    print(f"ANSWERS: {entry.get('answers')}")
    print("-" * 80)
    
    # 1. Show Ground Truth (Annotated Passage)
    print("GROUND TRUTH (Annotated Passage):")
    positive_ctxs = entry.get('positive_ctxs', [])
    if not positive_ctxs:
        print("  [No Ground Truth provided in dataset]")
    else:
        # Show annotated ground truths (usually 1, but sometimes list)
        for i, ctx in enumerate(positive_ctxs): 
            title = ctx.get('title', 'N/A')
            text = ctx.get('text', '')
            pid = ctx.get('passage_id', 'N/A')
            print(f"  [GT #{i+1}] ID: {pid} | Title: {title}")
            print(f"  Text: {text}") # Printing FULL text
            print()

    print("-" * 40)

    # 2. Show SEAL's Top 2 Retrieved Passages
    print("SEAL'S TOP 2 RETRIEVED PASSAGES:")
    retrieved_ctxs = entry.get('ctxs', [])
    if not retrieved_ctxs:
        print("  [No passages retrieved]")
    else:
        for i, ctx in enumerate(retrieved_ctxs[:2]): # Limit to Top 2
            title = ctx.get('title', 'N/A')
            text = ctx.get('text', '')
            score = ctx.get('score', 'N/A')
            pid = ctx.get('passage_id', 'N/A')
            
            print(f"  [Rank {i+1}] ID: {pid} | Score: {score} | Title: {title}")
            print(f"  Text: {text}") # Printing FULL text
            print()
            
    print("=" * 80)
    print("\n")

def main():
    json_path = "data/seal_output.json"
    csv_path = "generated_data/answer_location_analysis.csv"
    
    # 1. Load targets
    target_questions = load_target_questions(csv_path)
    
    if not target_questions:
        print("No targets found. Check CSV path and column names.")
        return

    # 2. Load JSON
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    print(f"Loading full data from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    if not isinstance(json_data, list):
        json_data = [json_data]

    # 3. Find matches and display
    print("Finding matches...\n")
    
    matches_found = 0
    total_targets = len(target_questions)
    
    for entry in json_data:
        q_text = entry.get('question', '').strip()
        
        if q_text in target_questions:
            matches_found += 1
            display_comparison(entry, matches_found, total_targets)

    print(f"Done. Displayed {matches_found} cases.")

if __name__ == "__main__":
    main()