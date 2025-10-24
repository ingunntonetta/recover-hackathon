"""
Exploratory Data Analysis for Recover Hackathon

Run this to understand the data distribution and patterns
"""

import sys
sys.path.append('.')

# Import without triggering Kaggle authentication
from dataset.hackathon import HackathonDataset
import polars as pl
import numpy as np


def analyze_dataset():
    print("="*60)
    print("RECOVER HACKATHON - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load training data - download=False since you have the files already
    print("\n1. Loading dataset...")
    dataset = HackathonDataset(split="train", download=False, seed=42)
    df = dataset.get_polars_dataframe()
    
    print(f"Total samples: {len(df)}")
    print(f"Number of unique projects: {df['project_id'].n_unique()}")
    
    # Room distribution
    print("\n2. Room Type Distribution:")
    room_counts = df.group_by("room_cluster").agg(pl.count()).sort("count", descending=True)
    print(room_counts)
    
    # Work operations per room
    print("\n3. Work Operations Statistics:")
    print(f"Mean operations per room: {df['work_operation'].count() / len(df):.2f}")
    
    ops_per_room = df.group_by(["project_id", "room"]).agg(
        pl.col("work_operation").n_unique().alias("num_operations")
    )
    print(f"Min operations in a room: {ops_per_room['num_operations'].min()}")
    print(f"Max operations in a room: {ops_per_room['num_operations'].max()}")
    print(f"Median operations in a room: {ops_per_room['num_operations'].median()}")
    
    # Sampling strategy analysis
    print("\n4. Sampling Strategy Analysis:")
    if "is_hidden" in df.columns:
        hidden_stats = df.group_by("is_hidden").agg(pl.count())
        print(hidden_stats)
        
        hidden_pct = (hidden_stats.filter(pl.col("is_hidden") == True)["count"][0] / 
                      df.height * 100)
        print(f"Percentage of hidden operations: {hidden_pct:.2f}%")
    
    # Most common work operations
    print("\n5. Top 20 Most Common Work Operations:")
    top_ops = (df.group_by("work_operation")
               .agg(pl.count().alias("frequency"))
               .sort("frequency", descending=True)
               .head(20))
    
    for row in top_ops.iter_rows(named=True):
        op_code = row["work_operation"]
        op_name = dataset.work_operations_dataset.code_to_wo.get(op_code, "Unknown")
        print(f"  {op_code}: {op_name} ({row['frequency']} occurrences)")
    
    # Room + operation combinations
    print("\n6. Room-Operation Patterns:")
    room_op_combos = df.group_by(["room_cluster", "work_operation"]).agg(pl.count())
    
    for room in ["bad", "kjøkken", "stue"]:
        print(f"\n  Most common in {room}:")
        room_ops = (room_op_combos.filter(pl.col("room_cluster") == room)
                   .sort("count", descending=True)
                   .head(5))
        for row in room_ops.iter_rows(named=True):
            op_name = dataset.work_operations_dataset.code_to_wo.get(row["work_operation"], "Unknown")
            print(f"    - {op_name} ({row['count']} times)")
    
    # Project metadata
    print("\n7. Project Metadata:")
    print(f"Number of unique insurance companies: {df['insurance_company'].n_unique()}")
    print(f"Year range: {df['case_creation_year'].min()} - {df['case_creation_year'].max()}")
    
    if "office_distance" in df.columns:
        print(f"Office distance - Mean: {df['office_distance'].mean():.2f} km")
        print(f"Office distance - Median: {df['office_distance'].median():.2f} km")
        print(f"Office distance - Max: {df['office_distance'].max():.2f} km")
    
    # Context size analysis
    print("\n8. Context Analysis (other rooms in project):")
    sample_with_context = [s for s in dataset if len(s.get("calculus", [])) > 0]
    if sample_with_context:
        context_sizes = [len(s["calculus"]) for s in sample_with_context]
        print(f"Average context rooms: {np.mean(context_sizes):.2f}")
        print(f"Max context rooms: {np.max(context_sizes)}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    analyze_dataset()