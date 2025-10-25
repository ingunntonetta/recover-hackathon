"""
Analyze Operation Co-occurrence Patterns
Find which operations tend to appear together
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

print("="*60)
print("OPERATION CO-OCCURRENCE ANALYSIS")
print("="*60)

# Load training data
print("\nLoading training data...")
train_df = pd.read_csv("data/train.csv")
print(f"✓ Loaded {len(train_df)} rows")

# Group by project and room to get operation sets
print("\nGrouping operations by room...")
grouped = train_df.groupby(['project_id', 'room']).agg({
    'work_operation_cluster_code': list,
    'work_operation_cluster_name': list
}).reset_index()

print(f"✓ Found {len(grouped)} unique rooms")

# Build co-occurrence matrix
print("\nBuilding co-occurrence matrix...")
num_ops = 388
cooccurrence = np.zeros((num_ops, num_ops), dtype=int)
operation_counts = Counter()

for ops in tqdm(grouped['work_operation_cluster_code'], desc="Processing rooms"):
    # Count individual operations
    for op in ops:
        if 0 <= op < num_ops:
            operation_counts[op] += 1
    
    # Count co-occurrences (pairs)
    for i, op1 in enumerate(ops):
        if 0 <= op1 < num_ops:
            for op2 in ops[i+1:]:
                if 0 <= op2 < num_ops:
                    cooccurrence[op1, op2] += 1
                    cooccurrence[op2, op1] += 1  # Symmetric

print("✓ Co-occurrence matrix built")

# Calculate conditional probabilities: P(op2 | op1)
print("\nCalculating conditional probabilities...")
conditional_prob = np.zeros((num_ops, num_ops), dtype=float)

for op1 in range(num_ops):
    if operation_counts[op1] > 0:
        for op2 in range(num_ops):
            if op1 != op2:
                # P(op2 | op1) = count(op1, op2) / count(op1)
                conditional_prob[op1, op2] = cooccurrence[op1, op2] / operation_counts[op1]

# Get operation names
print("\nLoading operation names...")
op_names = {}
for idx, row in train_df[['work_operation_cluster_code', 'work_operation_cluster_name']].drop_duplicates().iterrows():
    op_names[row['work_operation_cluster_code']] = row['work_operation_cluster_name']

# Find strongest correlations
print("\n" + "="*60)
print("TOP 50 STRONGEST OPERATION PAIRS")
print("="*60)

pairs = []
for op1 in range(num_ops):
    for op2 in range(op1+1, num_ops):
        if cooccurrence[op1, op2] > 5:  # At least 5 co-occurrences
            avg_prob = (conditional_prob[op1, op2] + conditional_prob[op2, op1]) / 2
            pairs.append((op1, op2, cooccurrence[op1, op2], avg_prob))

# Sort by probability
pairs.sort(key=lambda x: x[3], reverse=True)

print("\nFormat: Operation 1 ↔ Operation 2 | Co-occurrences | Probability\n")
for i, (op1, op2, count, prob) in enumerate(pairs[:50], 1):
    name1 = op_names.get(op1, f"Op{op1}")
    name2 = op_names.get(op2, f"Op{op2}")
    print(f"{i:2d}. {name1[:30]:30s} ↔ {name2[:30]:30s} | {count:5d} times | {prob:.2%}")

# Find operations that ALWAYS appear together
print("\n" + "="*60)
print("OPERATIONS THAT ALWAYS APPEAR TOGETHER (>90% probability)")
print("="*60)

always_together = []
for op1 in range(num_ops):
    for op2 in range(op1+1, num_ops):
        prob1 = conditional_prob[op1, op2]
        prob2 = conditional_prob[op2, op1]
        if prob1 > 0.9 and prob2 > 0.9 and cooccurrence[op1, op2] > 10:
            always_together.append((op1, op2, min(prob1, prob2)))

always_together.sort(key=lambda x: x[2], reverse=True)

for op1, op2, prob in always_together[:20]:
    name1 = op_names.get(op1, f"Op{op1}")
    name2 = op_names.get(op2, f"Op{op2}")
    print(f"{name1[:35]:35s} ↔ {name2[:35]:35s} | {prob:.1%}")

# Save co-occurrence matrix
print("\n" + "="*60)
print("Saving co-occurrence data...")
np.save("cooccurrence_matrix.npy", cooccurrence)
np.save("conditional_probability.npy", conditional_prob)
print("✓ Saved cooccurrence_matrix.npy")
print("✓ Saved conditional_probability.npy")

# Find removal → installation pairs
print("\n" + "="*60)
print("REMOVAL → INSTALLATION PAIRS")
print("="*60)

removal_install = []
for op1 in range(num_ops):
    name1 = op_names.get(op1, "")
    if any(word in name1.lower() for word in ['riv', 'fjern', 'demonter', 'remove']):
        for op2 in range(num_ops):
            name2 = op_names.get(op2, "")
            if any(word in name2.lower() for word in ['ny', 'monter', 'install', 'legg']):
                # Check if they reference similar things
                if cooccurrence[op1, op2] > 5:
                    prob = conditional_prob[op1, op2]
                    if prob > 0.3:
                        removal_install.append((op1, op2, prob, name1, name2))

removal_install.sort(key=lambda x: x[2], reverse=True)

for op1, op2, prob, name1, name2 in removal_install[:30]:
    print(f"{name1[:30]:30s} → {name2[:30]:30s} | {prob:.1%}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nKey insights to use:")
print("1. Strong co-occurrence pairs should be predicted together")
print("2. Removal operations often predict installation operations")
print("3. Use conditional_probability.npy as features in your model")