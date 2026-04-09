"""
TRAIN STRATEGY MAPPER

Learns how strategies change based on morphology.
"""
import sys
sys.path.insert(0, '/scratch/anshb3/ovla')

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

from ovla.strategy_mapper import StrategyMapper

print("="*70)
print("TRAINING STRATEGY MAPPER")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load data
with open('/scratch/anshb3/ovla/training_data/universal_strategy_samples.pkl', 'rb') as f:
    samples = pickle.load(f)

print(f"\nLoaded {len(samples)} samples")

# Create dataset (convert to numpy first!)
print("Converting to tensors...")
strategies = torch.FloatTensor(np.array([s['strategy'] for s in samples]))
morphologies = torch.FloatTensor(np.array([s['morphology'] for s in samples]))

print(f"Strategy shape: {strategies.shape}")
print(f"Morphology shape: {morphologies.shape}")

# For strategy mapper, we need source→target pairs
print("\nCreating strategy mapping pairs...")

training_pairs = []

# Group by primitive type
from collections import defaultdict
by_primitive = defaultdict(list)

for idx, sample in enumerate(samples):
    by_primitive[sample['primitive_type']].append(idx)

# Create pairs within same primitive
for primitive, indices in tqdm(by_primitive.items(), desc="Building pairs"):
    # Create all pairs
    for i in range(len(indices)):
        for j in range(len(indices)):
            if i != j:  # Different robots
                source_idx = indices[i]
                target_idx = indices[j]
                
                training_pairs.append({
                    'source_strategy': strategies[source_idx],
                    'source_morphology': morphologies[source_idx],
                    'target_morphology': morphologies[target_idx],
                    'target_strategy': strategies[target_idx],
                })

print(f"✓ Created {len(training_pairs)} training pairs")

# Split train/val
from sklearn.model_selection import train_test_split

train_pairs, val_pairs = train_test_split(
    training_pairs, test_size=0.1, random_state=42
)

print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

# Create model
model = StrategyMapper(
    strategy_dim=64,
    morphology_dim=256,
    hidden_dim=256
).to(device)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop
num_epochs = 50
batch_size = 128

best_val_loss = float('inf')

print("\nTraining...")

for epoch in range(num_epochs):
    
    # Training
    model.train()
    train_loss = 0.0
    
    # Shuffle
    np.random.shuffle(train_pairs)
    
    # Progress bar for batches
    num_batches = len(train_pairs) // batch_size
    
    pbar = tqdm(range(0, len(train_pairs), batch_size), 
                desc=f"Epoch {epoch+1}/{num_epochs}",
                leave=False)
    
    for i in pbar:
        batch = train_pairs[i:i+batch_size]
        
        source_strategy = torch.stack([p['source_strategy'] for p in batch]).to(device)
        source_morph = torch.stack([p['source_morphology'] for p in batch]).to(device)
        target_morph = torch.stack([p['target_morphology'] for p in batch]).to(device)
        target_strategy = torch.stack([p['target_strategy'] for p in batch]).to(device)
        
        optimizer.zero_grad()
        
        predicted_strategy, _ = model(source_strategy, source_morph, target_morph)
        
        loss = criterion(predicted_strategy, target_strategy)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    train_loss /= num_batches
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        val_pbar = tqdm(range(0, len(val_pairs), batch_size),
                       desc="Validation",
                       leave=False)
        
        for i in val_pbar:
            batch = val_pairs[i:i+batch_size]
            
            source_strategy = torch.stack([p['source_strategy'] for p in batch]).to(device)
            source_morph = torch.stack([p['source_morphology'] for p in batch]).to(device)
            target_morph = torch.stack([p['target_morphology'] for p in batch]).to(device)
            target_strategy = torch.stack([p['target_strategy'] for p in batch]).to(device)
            
            predicted_strategy, _ = model(source_strategy, source_morph, target_morph)
            
            loss = criterion(predicted_strategy, target_strategy)
            val_loss += loss.item()
            
            val_pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})
    
    val_loss /= (len(val_pairs) // batch_size)
    
    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '/scratch/anshb3/ovla/models/strategy_mapper_best.pt')
        print(f"✓ Epoch {epoch+1} - New best! Train: {train_loss:.6f}, Val: {val_loss:.6f}")
    else:
        print(f"  Epoch {epoch+1} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

print(f"\n✓ Training complete!")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Model saved to: /scratch/anshb3/ovla/models/strategy_mapper_best.pt")

print("\n" + "="*70)
