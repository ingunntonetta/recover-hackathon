"""
Baseline Solution for Recover Hackathon - FIXED VERSION
This implements a simple neural network baseline using the provided dataloader
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.hackathon import HackathonDataset
from dataset.collate import collate_fn
from metrics import normalized_rooms_score
from tqdm import tqdm
# At the top of the file:
from load_features import load_ticket_features


class BaselineModel(nn.Module):
    """Simple feedforward model with context aggregation"""
    
    def __init__(self, num_operations=388, num_room_types=11, hidden_dim=512):
        super().__init__()
        
        # Input: operations (388) + room type (11) + aggregated context
        self.room_encoder = nn.Sequential(
            nn.Linear(num_operations + num_room_types, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3)
        )
        
        # Context aggregation (simple mean pooling over context rooms)
        self.context_encoder = nn.Sequential(
            nn.Linear(num_operations + num_room_types, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        
        # Combine room and context
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Output layer
        self.classifier = nn.Linear(hidden_dim // 2, num_operations)
        
    def forward(self, x, context, context_mask):
        """
        Args:
            x: (batch_size, num_operations + num_room_types) - current room
            context: (batch_size, max_context_size, num_operations + num_room_types)
            context_mask: (batch_size, max_context_size) - valid context entries
        """
        # Convert to float if needed
        x = x.float()
        context = context.float()
        
        # Encode current room
        room_features = self.room_encoder(x)
        
        # Encode and aggregate context
        batch_size = context.shape[0]
        context_flat = context.reshape(-1, context.shape[-1])
        context_encoded = self.context_encoder(context_flat)
        context_encoded = context_encoded.reshape(batch_size, -1, context_encoded.shape[-1])
        
        # Masked mean pooling
        context_mask_expanded = context_mask.unsqueeze(-1).float()
        context_sum = (context_encoded * context_mask_expanded).sum(dim=1)
        context_count = context_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        context_aggregated = context_sum / context_count
        
        # Combine
        combined = torch.cat([room_features, context_aggregated], dim=1)
        features = self.combiner(combined)
        
        # Predict
        logits = self.classifier(features)
        return logits


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        x = batch["X"].to(device)
        y = batch["Y"].to(device).float()
        context = batch["context"].to(device)
        context_mask = batch["context_mask"].to(device)
        
        optimizer.zero_grad()
        outputs = model(x, context, context_mask)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device, threshold=0.3):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            x = batch["X"].to(device)
            y = batch["Y"].to(device)
            context = batch["context"].to(device)
            context_mask = batch["context_mask"].to(device)
            
            outputs = model(x, context, context_mask)
            probs = torch.sigmoid(outputs)
            
            # Convert to predictions
            preds = (probs > threshold).cpu().numpy()
            targets = y.cpu().numpy()
            
            # Convert to list of lists of operation codes
            for pred, target in zip(preds, targets):
                pred_codes = [i+1 for i, val in enumerate(pred) if val == 1]
                target_codes = [i+1 for i, val in enumerate(target) if val == 1]
                all_preds.append(pred_codes)
                all_targets.append(target_codes)
    
    score = normalized_rooms_score(all_preds, all_targets)
    return score


def main():
    # Configuration
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = HackathonDataset(split="train", download=False, seed=42)
    val_dataset = HackathonDataset(split="val", download=False, seed=42)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = BaselineModel().to(device)
    
    # Loss and optimizer
    # Use BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (removed verbose parameter for compatibility)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Training loop
    best_score = -float('inf')
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate every 2 epochs to save time
        if (epoch + 1) % 2 == 0 or epoch == 0:
            val_score = validate(model, val_loader, device)
            print(f"Validation Score: {val_score:.4f}")
            
            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_score)
            new_lr = optimizer.param_groups[0]['lr']
            
            if new_lr != old_lr:
                print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Save best model
            if val_score > best_score:
                best_score = val_score
                torch.save(model.state_dict(), "best_model.pth")
                print(f"✓ New best score! Model saved.")
        
        # Reshuffle training data with new sampling strategy every 3 epochs
        if (epoch + 1) % 3 == 0 and epoch > 0:
            print("Reshuffling training data...")
            train_dataset.shuffle()
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Best validation score: {best_score:.4f}")
    print(f"Model saved to: best_model.pth")
    print(f"{'='*60}")
    
    # Final validation with best model
    print("\nRunning final validation with best model...")
    model.load_state_dict(torch.load("best_model.pth"))
    final_score = validate(model, val_loader, device)
    print(f"\nFinal validation score: {final_score:.4f}")
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Run: python3 generate_submission.py")
    print("2. Submit to Kaggle!")
    print("="*60)


if __name__ == "__main__":
    main()