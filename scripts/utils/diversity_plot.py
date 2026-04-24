import json
import matplotlib.pyplot as plt

paths = {
    'SEAL (NQ)': '/mnt/ssd/school_career/thesis_stuff/Thesis/generated_data/seal_nq/repetitive_tokens_results.json',
    'MINDER (NQ)': '/mnt/ssd/school_career/thesis_stuff/Thesis/generated_data/minder_nq/repetitive_tokens_results.json',
    'MINDER (MSMARCO)': '/mnt/ssd/school_career/thesis_stuff/Thesis/generated_data/minder_msmarco/repetitive_tokens_results.json'
}

colors = {
    'SEAL (NQ)': '#1f77b4',
    'MINDER (NQ)': '#ff7f0e',
    'MINDER (MSMARCO)': '#2ca02c'
}

plt.figure(figsize=(12, 8))

for label, path in paths.items():
    with open(path, 'r') as f:
        data = json.load(f)
        
    x = [(d['diversity_min'] + d['diversity_max']) / 2 for d in data['deciles']]
    y = [d['hits_at_1'] / 100 for d in data['deciles']]
    n_vals = [d['count'] for d in data['deciles']]

    plt.plot(x, y, marker='o', color=colors[label], label=label)
    
    for xi, yi, ni in zip(x, y, n_vals):
        plt.text(xi, yi + 0.015, f'n={ni}', color=colors[label], fontsize=8, fontweight='bold', ha='center', va='bottom')

plt.title('H@1 Rate vs Diversity', fontsize=16)
plt.xlabel('Diversity (Decile Midpoint)', fontweight='bold', fontsize=16)
plt.ylabel('H@1 Rate', fontweight='bold', fontsize=16)
plt.xlim(0.1, 1.0)
plt.ylim(0.0, 1.0)
plt.grid(True, linestyle='--', color='lightgray')
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('h1_rate_vs_diversity.png')