import sys
import ijson
from collections import Counter

def main(datapath="data/seal_output.json"):
    print(f"Checking for duplicate passage IDs in {datapath}...")
    
    duplicate_count = 0
    total_entries = 0
    
    try:
        with open(datapath, "rb") as f:
            for i, item in enumerate(ijson.items(f, "item")):
                total_entries += 1
                ctxs = item.get("ctxs", [])
                if not ctxs:
                    continue
                
                # Extract passage IDs for this query
                pids = [ctx.get("passage_id") for ctx in ctxs if ctx.get("passage_id")]
                
                # Check for duplicates
                counts = Counter(pids)
                duplicates = [pid for pid, count in counts.items() if count > 1]
                
                if duplicates:
                    duplicate_count += 1
                    print(f"Found duplicates in entry {i} (Question: '{item.get('question', '')[:50]}...'):")
                    for pid in duplicates:
                        print(f"  Passage ID '{pid}' appears {counts[pid]} times")

        print(f"\nScanned {total_entries} entries.")
        if duplicate_count == 0:
            print("No duplicate passage IDs found within any single 'ctxs' list.")
        else:
            print(f"Found duplicates in {duplicate_count} entries.")

    except FileNotFoundError:
        print(f"File not found: {datapath}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/seal_output.json"
    main(path)
