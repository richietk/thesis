import sys
import ijson
from utils.utils import get_ground_truth_ids

def find_zero_hits(datapath="data/seal_output.json"):
    print(f"Searching for zero-hit queries in {datapath}...")
    
    zero_hit_queries = []
    total_processed = 0
    
    try:
        with open(datapath, "rb") as f:
            for item in ijson.items(f, "item"):
                total_processed += 1
                
                # Get gold IDs
                gold_ids = get_ground_truth_ids(item)
                
                # Get retrieved IDs
                ctxs = item.get("ctxs", [])
                retrieved_ids = [ctx.get("passage_id") for ctx in ctxs if ctx.get("passage_id")]
                
                # Check for any intersection
                has_hit = False
                for rid in retrieved_ids:
                    base_rid = rid.split('...')[0]
                    if rid in gold_ids or base_rid in gold_ids:
                        has_hit = True
                        break
                
                if not has_hit:
                    zero_hit_queries.append({
                        'question': item.get('question'),
                        'gold_count': len(gold_ids),
                        'retrieved_count': len(retrieved_ids)
                    })
                    
        if total_processed == 0:
            print("No queries found in the file.")
            return

        pct_zero_hits = (len(zero_hit_queries) / total_processed) * 100
        
        print(f"\nProcessed {total_processed} queries.")
        print(f"Zero-hit queries: {len(zero_hit_queries)} ({pct_zero_hits:.2f}% of total)")
        
        if zero_hit_queries:
            # Sort by gold count descending for 'most gold'
            most_gold = sorted(zero_hit_queries, key=lambda x: x['gold_count'], reverse=True)[:3]
            
            # Sort by gold count ascending for 'least gold' (excluding 0 if possible, though gold_ids usually > 0)
            least_gold = sorted([q for q in zero_hit_queries if q['gold_count'] > 0], key=lambda x: x['gold_count'])[:3]
            
            # If all had 0 gold, fallback
            if not least_gold and zero_hit_queries:
                least_gold = sorted(zero_hit_queries, key=lambda x: x['gold_count'])[:3]

            print(f"\nTop 3 Zero-Hit Queries with MOST Gold Contexts:")
            for i, q in enumerate(most_gold, 1):
                print(f"{i}. Question: '{q['question']}'")
                print(f"   (Gold: {q['gold_count']}, Retrieved: {q['retrieved_count']})")
                
            print(f"\nTop 3 Zero-Hit Queries with LEAST Gold Contexts:")
            for i, q in enumerate(least_gold, 1):
                print(f"{i}. Question: '{q['question']}'")
                print(f"   (Gold: {q['gold_count']}, Retrieved: {q['retrieved_count']})")

    except FileNotFoundError:
        print(f"File not found: {datapath}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/seal_output.json"
    find_zero_hits(path)
