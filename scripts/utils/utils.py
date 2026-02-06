"""
Common utility functions for analysis scripts.
"""

import os
import re
import ast
import ijson
from typing import Dict, List, Any
from transformers import GPT2TokenizerFast

# Initialize tokenizer once for efficiency
_tokenizer = None

def get_tokenizer():
    """Get or initialize the GPT2 tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return _tokenizer


def get_dataset_name(datapath: str) -> str:
    """Extract dataset name (seal or minder) from datapath."""
    if "minder" in datapath.lower():
        return "minder"
    elif "seal" in datapath.lower():
        return "seal"
    else:
        return os.path.splitext(os.path.basename(datapath))[0]


def strip_ngram_markers(ngram: str, datapath: str) -> str:
    """Strip pseudoquery markers from ngrams if using Minder data."""
    if "minder_output.json" in datapath:
        ngram = ngram.replace(" ||", "").strip()
    return ngram


def strip_pseudoqueries(text: str, datapath: str) -> str:
    """Strip pseudoquery markers from text if using Minder data."""
    if "minder_output.json" in datapath:
        # Remove || ... @@ patterns
        text = re.sub(r'\|\|[^@]*@@', '', text)
    return text


def parse_ngrams(keys_str):
    """Parse n-gram keys from string or list format, handling Decimal objects."""
    if not keys_str:
        return []
    try:
        # If it's already a list, use it directly
        if isinstance(keys_str, list):
            keys_list = keys_str
        else:
            # Try to parse string format
            keys_list = ast.literal_eval(keys_str)

        # Convert Decimal to float for all score values
        result = []
        for ngram, freq, score in keys_list:
            result.append((ngram, int(freq), float(score)))
        return result
    except:
        return []


def normalize_text(text: str) -> str:
    """Normalize text for matching: lowercase, strip whitespace."""
    return text.lower().strip()


def answer_in_text_substring(answer: str, text: str) -> bool:
    """
    OLD METHOD: Check if answer appears in text (simple substring, case-insensitive).
    This is kept for comparison purposes to show false positives.
    """
    return normalize_text(answer) in normalize_text(text)


def answer_in_text(answer: str, text: str) -> bool:
    """
    Check if answer appears in text using tokenized matching.

    Tokenizes both answer and text, then checks if answer tokens appear
    as a contiguous subsequence in the text tokens (case-insensitive).

    Handles both space-prefixed and non-space-prefixed versions since
    BPE tokenization is sensitive to leading whitespace.

    Args:
        answer: The answer string to search for
        text: The text to search in

    Returns:
        True if answer tokens found in text, False otherwise
    """
    if not answer or not text:
        return False

    # Normalize both strings
    answer_norm = normalize_text(answer)
    text_norm = normalize_text(text)

    # Get tokenizer
    tokenizer = get_tokenizer()

    # Tokenize text
    text_tokens = tokenizer.encode(text_norm, add_special_tokens=False)

    # Try both with and without leading space (BPE is whitespace-sensitive)
    answer_variants = [
        answer_norm,           # "paul"
        " " + answer_norm,     # " paul"
    ]

    for answer_variant in answer_variants:
        answer_tokens = tokenizer.encode(answer_variant, add_special_tokens=False)

        if not answer_tokens:
            continue

        answer_len = len(answer_tokens)

        # Sliding window to find answer token sequence
        for i in range(len(text_tokens) - answer_len + 1):
            if text_tokens[i:i + answer_len] == answer_tokens:
                return True

    return False


def get_ground_truth_ids(query_data: Dict) -> set:
    """Extract ground-truth passage IDs from query data."""
    gold_ids = set()
    if 'positive_ctxs' in query_data:
        for ctx in query_data['positive_ctxs']:
            gold_ids.add(ctx.get('passage_id', '').split('...')[0])
    return gold_ids


def stream_data(file_path: str):
    """Stream data from JSON file using ijson."""
    with open(file_path, 'rb') as f:
        parser = ijson.items(f, 'item')
        for entry in parser:
            yield entry


class DecileTablePrinter:
    """
    Utility class for printing consistent decile analysis tables.

    Example usage:
        printer = DecileTablePrinter(
            metric_name="Unigram Fraction",
            columns=["Top-1", "Top-2", "Top-10"]
        )

        for decile_num, bin_data in enumerate(decile_bins, start=1):
            printer.add_row(
                decile=decile_num,
                range_min=bin_data['min'],
                range_max=bin_data['max'],
                values=[bin_data['top1'], bin_data['top2'], bin_data['top10']],
                count=bin_data['count']
            )

        printer.print_table()
    """

    def __init__(self, metric_name: str, columns: List[str], include_count: bool = True):
        """
        Initialize the decile table printer.

        Args:
            metric_name: Name of the metric being analyzed (e.g., "Unigram Fraction")
            columns: List of column names for the data values
            include_count: Whether to include a Count column
        """
        self.metric_name = metric_name
        self.columns = columns
        self.include_count = include_count
        self.rows = []

    def add_row(self, decile: int, range_min: float, range_max: float,
                values: List[float], count: int = None):
        """
        Add a row to the table.

        Args:
            decile: Decile number (1-10)
            range_min: Minimum value in the range
            range_max: Maximum value in the range
            values: List of values for each column (should match number of columns)
            count: Number of items in this decile (optional)
        """
        if len(values) != len(self.columns):
            raise ValueError(f"Expected {len(self.columns)} values, got {len(values)}")

        self.rows.append({
            'decile': decile,
            'range_min': range_min,
            'range_max': range_max,
            'values': values,
            'count': count
        })

    def print_table(self):
        """Print the formatted decile table."""
        # Determine column widths
        metric_width = max(len(self.metric_name) + 6, 18)  # "range" label
        col_width = 8

        # Build header
        header_parts = [f"{self.metric_name} range".ljust(metric_width)]
        for col in self.columns:
            header_parts.append(col.rjust(col_width))
        if self.include_count:
            header_parts.append("Count".rjust(8))

        header = " | ".join(header_parts)
        separator = "-" * len(header)

        # Print table
        print(header)
        print(separator)

        for row in self.rows:
            # Format range
            range_str = f"{row['range_min']:.2f}–{row['range_max']:.2f}"
            parts = [range_str.ljust(metric_width)]

            # Format values (assume percentages)
            for val in row['values']:
                if isinstance(val, (int, float)):
                    parts.append(f"{val:6.2f}%".rjust(col_width))
                else:
                    parts.append(str(val).rjust(col_width))

            # Add count if needed
            if self.include_count and row['count'] is not None:
                parts.append(f"{row['count']}".rjust(8))

            print(" | ".join(parts))

        print(separator)


def calculate_retrieval_metrics(retrieved_ids, gold_ids):
    """
    Calculate precision@1, hits@10, and R-precision for retrieval results.

    Args:
        retrieved_ids: List of passage IDs from SEAL 'ctxs' (in retrieval order)
        gold_ids: Set of ground truth passage IDs from 'positive_ctxs'

    Returns:
        dict: Contains "precision_at_1", "hits_at_10", and "r_precision"
    """
    if not gold_ids:
        return {"precision_at_1": 0.0, "hits_at_10": 0.0, "r_precision": 0.0}

    if not retrieved_ids:
        return {"precision_at_1": 0.0, "hits_at_10": 0.0, "r_precision": 0.0}

    # Precision@1: 1 if top retrieved document is relevant, else 0
    p1 = 1.0 if retrieved_ids[0] in gold_ids else 0.0

    # Hits@10: 1 if any of top-10 retrieved documents are relevant, else 0
    top_10 = retrieved_ids[:10]
    hits_10 = 1.0 if any(pid in gold_ids for pid in top_10) else 0.0

    # R-Precision: precision at R where R = number of relevant documents
    R = len(gold_ids)
    top_R = retrieved_ids[:R]
    found_in_R = sum(1 for pid in top_R if pid in gold_ids)
    r_prec = found_in_R / R if R > 0 else 0.0

    return {
        "precision_at_1": p1,
        "hits_at_10": hits_10,
        "r_precision": r_prec
    }
