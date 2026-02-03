"""
SEAL Failure Analysis Pipeline
==============================
Runs all analysis scripts in logical order.

Input: data/seal_output.json (6,515 queries with SEAL retrieval results)
"""

import subprocess
import sys
import os

def run(script, description):
    """Run a script with a description."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {script}")
    print(f"PURPOSE: {description}")
    print('='*60)
    
    if not os.path.exists(script):
        print(f"ERROR: {script} not found, skipping...")
        return
    
    subprocess.run([sys.executable, script])


def main():
    # Check data exists
    if not os.path.exists("data/seal_output.json"):
        print("ERROR: data/seal_output.json not found")
        sys.exit(1)
    
    print("="*60)
    print("SEAL FAILURE ANALYSIS PIPELINE")
    print("="*60)
    
    # Step 1: Understand the data structure
    run("print_out_json_structure.py", 
        "Shows the structure of the JSON data")
    
    # Step 2: See sample entries
    run("sample_output.py", 
        "Prints sample entries to understand the data format")
    
    # Step 3: Verify passage chunking (100-word chunks)
    run("verify_passage_length.py", 
        "Confirms passages are 100-word chunks (not truncated)")
    
    # Step 4: Core analysis - PP/PN/NP/NN classification
    run("seal_analysis_v1.py", 
        "Classifies queries by retrieval outcome and computes n-gram statistics")
    # Output: seal_unified_analysis.csv
    
    # Step 5: Answer location analysis
    run("answer_location_analysis.py", 
        "Categorizes: GT found (76.1%) / answer elsewhere (5.7%) / not found (18.2%)")
    # Output: answer_location_analysis.csv
    
    # Step 6: Find specific example queries (Paul, Gorsuch, etc.)
    run("find_specific_queries.py", 
        "Finds and displays specific queries for thesis examples")
    
    # Step 7: Examine 'answer in different passage' cases
    run("examine_answer_in_diff_psg.py", 
        "Shows full comparison for the 369 ambiguous cases")
    
    # Step 8: LLM-as-a-judge classification (requires API key)
    print(f"\n{'='*60}")
    print("SKIPPING: llm_judge_concept.py")
    print("PURPOSE: LLM classification of failures (requires Cohere API key)")
    print("Run manually: python llm_judge_concept.py")
    print('='*60)
    
    # Done
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print("="*60)
    print("""
Outputs:
  - generated_data/seal_unified_analysis.csv      : N-gram statistics per query
  - generated_data/answer_location_analysis.csv   : Answer location classification

Run manually if needed:
  - python llm_judge_concept.py    : LLM failure classification (needs API)
""")


if __name__ == "__main__":
    main()