"""
Generate predictions for test set and create submission file - INDEXING FIXED
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.hackathon import HackathonDataset
from dataset.collate import collate_fn
from tqdm import tqdm


class BaselineModel(nn.Module):
    """Copy of model from baseline_solution.py - WITH FLOAT CONVERSION"""
    
    def __init__(self, num_operations=388, num_room_types=11, hidden_dim=512):
        super().__init__()
        
        self.room_encoder = nn.Sequential(
            nn.Linear(num_operations + num_room_types, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3)
        )
        
        self.context_encoder = nn.Sequential(
            nn.Linear(num_operations + num_room_types, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, num_operations)
        
    def forward(self, x, context, context_mask):
        # Convert to float if needed
        x = x.float()
        context = context.float()
        
        room_features = self.room_encoder(x)
        
        batch_size = context.shape[0]
        context_flat = context.reshape(-1, context.shape[-1])
        context_encoded = self.context_encoder(context_flat)
        context_encoded = context_encoded.reshape(batch_size, -1, context_encoded.shape[-1])
        
        context_mask_expanded = context_mask.unsqueeze(-1).float()
        context_sum = (context_encoded * context_mask_expanded).sum(dim=1)
        context_count = context_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        context_aggregated = context_sum / context_count
        
        combined = torch.cat([room_features, context_aggregated], dim=1)
        features = self.combiner(combined)
        
        logits = self.classifier(features)
        return logits


def generate_predictions(model_path="best_model.pth", threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = HackathonDataset(split="test", download=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model
    print("Loading model...")
    model = BaselineModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✓ Model loaded successfully")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting")):
            x = batch["X"].to(device)
            context = batch["context"].to(device)
            context_mask = batch["context_mask"].to(device)
            
            outputs = model(x, context, context_mask)
            probs = torch.sigmoid(outputs)
            
            # Convert to predictions
            preds = (probs > threshold).cpu().numpy()
            
            # Get IDs from test dataset
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + len(preds)
            
            for i, pred in enumerate(preds):
                sample_idx = start_idx + i
                if sample_idx < len(test_dataset):
                    sample = test_dataset[sample_idx]
                    sample_id = sample["id"]
                    
                
                    pred_codes = [j for j, val in enumerate(pred) if val == 1]
                    predictions[sample_id] = pred_codes
    
    print(f"\n✓ Generated predictions for {len(predictions)} samples")
    
    # Statistics
    num_ops_per_room = [len(ops) for ops in predictions.values()]
    print(f"\nPrediction statistics:")
    print(f"  Average operations per room: {sum(num_ops_per_room) / len(num_ops_per_room):.2f}")
    print(f"  Min operations predicted: {min(num_ops_per_room)}")
    print(f"  Max operations predicted: {max(num_ops_per_room)}")
    print(f"  Rooms with 0 predictions: {sum(1 for x in num_ops_per_room if x == 0)}")
    
    # Create submission
    print("\nCreating submission file...")
    test_dataset.create_submission(predictions)
    print("✓ Done!")
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Find your submission file in submissions/ folder")
    print("2. Go to: https://www.kaggle.com/competitions/hackathon-recover-x-cogito/submit")
    print("3. Upload and submit!")
    print("="*60)


if __name__ == "__main__":
    generate_predictions()