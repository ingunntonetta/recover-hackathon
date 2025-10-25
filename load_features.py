# load_features.py
import pandas as pd
import numpy as np
import torch

def load_ticket_features():
    """Load ticket counts for each operation"""
    tickets_df = pd.read_csv("data/tickets.csv")
    
    # Create array: index = operation code, value = ticket count
    ticket_counts = np.zeros(388, dtype=np.float32)
    
    for _, row in tickets_df.iterrows():
        op_code = row['work_operation_cluster_code']
        n_tickets = row['n_tickets']
        if 0 <= op_code < 388:
            ticket_counts[op_code] = n_tickets
    
    # Normalize to 0-1 range
    if ticket_counts.max() > 0:
        ticket_counts = ticket_counts / ticket_counts.max()
    
    print(f"✓ Loaded {(ticket_counts > 0).sum()} operations with ticket data")
    return torch.from_numpy(ticket_counts)