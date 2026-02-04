"""
Inspect 'Answer in Different Passage' Cases (Full Text)
=======================================================
1. Reads 'answer_location_analysis.csv' to identify queries categorized 
   as 'answer_in_different_passage'.
2. Loads the full data from 'data/seal_output.json'.
3. Prints the full comparison (Question, GT, Top 2 Retrieved) without text truncation.
"""

import ijson
import csv
import os
import re

def strip_pseudoqueries(text: str, datapath: str) -> str:
    """Strip pseudoquery markers from text if using Minder data."""
    if "minder_output.json" in datapath:
        # Remove || ... @@ patterns
        text = re.sub(r'\|\|[^@]*@@', '', text)
    return text


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

def display_comparison(entry, index, total, datapath=""):
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
            text = strip_pseudoqueries(ctx.get('text', ''), datapath)
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
            text = strip_pseudoqueries(ctx.get('text', ''), datapath)
            score = ctx.get('score', 'N/A')
            pid = ctx.get('passage_id', 'N/A')

            print(f"  [Rank {i+1}] ID: {pid} | Score: {score} | Title: {title}")
            print(f"  Text: {text}") # Printing FULL text
            print()
            
    print("=" * 80)
    print("\n")

def main(datapath="data/seal_output.json"):
    import sys
    import re

    script_name = "example_those_in_diff_psg"
    print(f"running {script_name}")

    try:
        # Determine dataset name
        dataset_name = "seal"
        if "minder" in datapath.lower():
            dataset_name = "minder"
        elif "seal" in datapath.lower():
            dataset_name = "seal"

        # Create output directory and log file
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f"{script_name}_log.txt")

        # Redirect stdout to log file
        original_stdout = sys.stdout
        sys.stdout = open(log_file, 'w', encoding='utf-8')

        json_path = datapath
        csv_path = f"generated_data/{dataset_name}/answer_location_analysis_{dataset_name}.csv"

        # 1. Load targets
        target_questions = load_target_questions(csv_path)

        if not target_questions:
            print("No targets found. Check CSV path and column names.")
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"error: running {script_name} No targets found")
            return

        # 2. Stream JSON
        if not os.path.exists(json_path):
            print(f"Error: JSON file not found at {json_path}")
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"error: running {script_name} JSON file not found")
            return

        print(f"Streaming data from {json_path}...")

        # 3. Find matches and display
        print("Finding matches...\n")

        matches_found = 0
        total_targets = len(target_questions)

        with open(json_path, "rb") as f:
            parser = ijson.items(f, 'item')

            for entry in parser:
                q_text = entry.get('question', '').strip()

                if q_text in target_questions:
                    matches_found += 1
                    display_comparison(entry, matches_found, total_targets, datapath)

        print(f"Done. Displayed {matches_found} cases.")

        # Restore stdout
        sys.stdout.close()
        sys.stdout = original_stdout

        print(f"success running {script_name}")

    except Exception as e:
        # Restore stdout in case of error
        if sys.stdout != original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout
        print(f"error: running {script_name} {e}")
        raise

if __name__ == "__main__":
    import sys
    datapath = sys.argv[1] if len(sys.argv) > 1 else "data/seal_output.json"
    main(datapath)