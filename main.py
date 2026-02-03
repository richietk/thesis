"""
SEAL Thesis Analysis Pipeline - Master Script

This script runs all analysis code for the thesis in sequence.
Each analysis module is executed independently with error handling.

The pipeline analyzes both SEAL and MINDER output files:
    - data/seal_output.json
    - data/minder_output.json

Usage:
    python main.py

All analyses are run on both data files automatically.
"""

import sys
from pathlib import Path
import traceback
from datetime import datetime

OUTPUT_DIR = Path("generated_data")
OUTPUT_DIR.mkdir(exist_ok=True)

ANALYSIS_MODULES = [
    ("answer_location_analysis", "Answer Location Analysis"),
    ("example_those_in_diff_psg", "Answer in Different Passage - Full Text Examples"),
    ("seal_analysis_prelim", "Preliminary SEAL Analysis with N-gram Examples"),
    ("ngram_count_analysis", "N-gram Count Distribution Analysis"),
    ("single_ngram_dominance", "Single N-gram Dominance Analysis"),
    ("single_ngram_test", "Single N-gram Dominance vs Success Rate Test"),
    ("ngram_length_bias", "N-gram Length Bias Analysis"),
    ("ngram_length_test", "Unigram Fraction vs Success Rate Test"),
    ("nonspecific_highfreq_ngrams", "N-gram Frequency Analysis"),
    ("query_n_gram_overlap", "Query-N-gram Overlap Analysis"),
    ("answer_coverage", "Answer Coverage in N-grams"),
    ("repetitive_tokens", "Token Diversity Analysis"),
    ("article_concentration", "Article Diversity Analysis"),
    ("scoring_failure", "Scoring Failure Analysis"),
    ("analyze_length_bias", "Document Length Bias Analysis"),
    ("analyze_failure_modes", "Comprehensive Failure Mode Analysis"),
    ("llm_judge_answer_verification", "LLM Judge: Answer Verification"),
    ("llm_judge_concept", "LLM Judge: Failure Classification"),
]


def run_analysis(module_name, description, datapath):
    print(f"Running: {description}")
    try:
        module = __import__(f'scripts.{module_name}', fromlist=[module_name])
        if hasattr(module, 'main'):
            module.main(datapath)
        else:
            print(f"Warning: No main() function found in {module_name}")
        print(f"Completed: {description}")
        return True
    except Exception as e:
        print(f"FAILED: {description}")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return False


def main():
    start_time = datetime.now()

    data_files = [
        "data/seal_output.json",
        "data/minder_output.json"
    ]

    print("\nSEAL THESIS ANALYSIS PIPELINE")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {OUTPUT_DIR.absolute()}")
    print(f"Total Analyses: {len(ANALYSIS_MODULES)}")
    print(f"Data Files: {', '.join(data_files)}\n")

    for datapath in data_files:
        data_file = Path(datapath)
        if not data_file.exists():
            print(f"ERROR: Data file not found at {data_file}")
            print("Please ensure the output data is available.")
            sys.exit(1)
        print(f"Data file found: {data_file}")

    all_results = {}

    for datapath in data_files:
        print(f"\nANALYZING: {datapath}\n")

        results = {
            'success': [],
            'failed': [],
            'skipped': []
        }

        for i, (module_name, description) in enumerate(ANALYSIS_MODULES, 1):
            print(f"\nAnalysis {i}/{len(ANALYSIS_MODULES)}: {description}")

            module_file = Path(f"scripts/{module_name}.py")
            if not module_file.exists():
                print(f"Skipping: Module file not found: {module_file}")
                results['skipped'].append((module_name, description))
                continue

            success = run_analysis(module_name, description, datapath)

            if success:
                results['success'].append((module_name, description))
            else:
                results['failed'].append((module_name, description))

        all_results[datapath] = results

    end_time = datetime.now()
    duration = end_time - start_time

    print("\nPIPELINE COMPLETE")
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")

    total_success = sum(len(all_results[df]['success']) for df in all_results)
    total_failed = sum(len(all_results[df]['failed']) for df in all_results)
    total_skipped = sum(len(all_results[df]['skipped']) for df in all_results)

    print(f"\nOverall Results Summary:")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_failed}")
    print(f"Skipped: {total_skipped}")

    for datapath, results in all_results.items():
        print(f"\n{datapath}:")
        print(f"Successful: {len(results['success'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Skipped: {len(results['skipped'])}")

        if results['failed']:
            print("Failed Analyses:")
            for module_name, description in results['failed']:
                print(f"  {description}")

    print(f"\nAll results saved to: {OUTPUT_DIR.absolute()}")

    if total_failed > 0:
        print(f"\nWarning: {total_failed} analyses failed. Please review the errors above.")
        sys.exit(1)
    else:
        print("\nAll analyses completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error in pipeline:")
        print(f"{str(e)}")
        traceback.print_exc()
        sys.exit(1)
