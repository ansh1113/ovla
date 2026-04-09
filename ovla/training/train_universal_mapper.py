"""
TRAIN UNIVERSAL SEMANTIC MAPPER - FIXED FOR VARIABLE DOF
"""
import sys
sys.path.insert(0, '/scratch/anshb3/ovla')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path

from ovla.universal_semantic_mapper import UniversalSemanticMapper, extract_urdf_graph_data

print("="*70)
print("TRAINING UNIVERSAL SEMANTIC MAPPER")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# ============================================================================
# DATASET
# ============================================================================

class SemanticPrimitiveDataset(Dataset):
    """Dataset for semantic primitive training"""
    
    def __init__(self, data_path: str, max_samples: int = None):
        with open(data_path, 'rb') as f:
            self.samples = pickle.load(f)
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        # Cache URDF data
        self.urdf_cache = {}
        unique_urdfs = set(s['urdf_path'] for s in self.samples)
        
        print(f"\nCaching URDF graph data for {len(unique_urdfs)} robots...")
        for urdf_path in unique_urdfs:
            try:
                self.urdf_cache[urdf_path] = extract_urdf_graph_data(urdf_path)
            except Exception as e:
                print(f"  ✗ Failed: {Path(urdf_path).stem}")
        
        self.samples = [s for s in self.samples if s['urdf_path'] in self.urdf_cache]
        print(f"✓ Dataset ready: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        urdf_data = self.urdf_cache[sample['urdf_path']]
        
        semantic_fingerprint = torch.FloatTensor(sample['semantic_fingerprint'])
        motion_type_embedding = torch.FloatTensor(sample['motion_type_embedding'])
        node_features = torch.FloatTensor(urdf_data['node_features'])
        adjacency = torch.FloatTensor(urdf_data['adjacency'])
        
        # Target delta
        target_actions = sample['target_state'] - sample['current_state']
        
        # Pad to 32
        max_dof = 32
        if len(target_actions) < max_dof:
            padding = np.zeros(max_dof - len(target_actions))
            target_actions = np.concatenate([target_actions, padding])
        else:
            target_actions = target_actions[:max_dof]
        
        target_actions = torch.FloatTensor(target_actions)
        
        return {
            'semantic_fingerprint': semantic_fingerprint,
            'motion_type_embedding': motion_type_embedding,
            'node_features': node_features,
            'adjacency': adjacency,
            'target_actions': target_actions,
            'dof': urdf_data['dof']
        }

# ============================================================================
# CUSTOM COLLATE FUNCTION
# ============================================================================

def custom_collate(batch):
    """
    Custom collate to handle variable-sized node_features and adjacency matrices.
    
    Strategy: Pad to max size in batch
    """
    # Find max nodes in this batch
    max_nodes = max(b['node_features'].shape[0] for b in batch)
    
    # Pad each sample
    semantic_fps = []
    motion_embs = []
    node_feats = []
    adjacencies = []
    targets = []
    dofs = []
    
    for b in batch:
        semantic_fps.append(b['semantic_fingerprint'])
        motion_embs.append(b['motion_type_embedding'])
        
        # Pad node features
        curr_nodes = b['node_features'].shape[0]
        if curr_nodes < max_nodes:
            padding = torch.zeros(max_nodes - curr_nodes, b['node_features'].shape[1])
            node_feat = torch.cat([b['node_features'], padding], dim=0)
        else:
            node_feat = b['node_features']
        node_feats.append(node_feat)
        
        # Pad adjacency
        if curr_nodes < max_nodes:
            padded_adj = torch.zeros(max_nodes, max_nodes)
            padded_adj[:curr_nodes, :curr_nodes] = b['adjacency']
        else:
            padded_adj = b['adjacency']
        adjacencies.append(padded_adj)
        
        targets.append(b['target_actions'])
        dofs.append(b['dof'])
    
    return {
        'semantic_fingerprint': torch.stack(semantic_fps),
        'motion_type_embedding': torch.stack(motion_embs),
        'node_features': torch.stack(node_feats),
        'adjacency': torch.stack(adjacencies),
        'target_actions': torch.stack(targets),
        'dof': torch.tensor(dofs)
    }

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*70)
print("LOADING DATASET")
print("="*70)

dataset = SemanticPrimitiveDataset(
    '/scratch/anshb3/ovla/training_data/semantic_primitives/primitive_dataset.pkl'
)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

print(f"\nSplit:")
print(f"  Train: {len(train_dataset):,} samples")
print(f"  Val:   {len(val_dataset):,} samples")

# USE CUSTOM COLLATE
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)

# ============================================================================
# MODEL
# ============================================================================

print("\n" + "="*70)
print("INITIALIZING MODEL")
print("="*70)

model = UniversalSemanticMapper(
    max_dof=32,
    semantic_dim=256,
    joint_dim=256,
    num_attention_heads=8
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n✓ Model initialized: {total_params:,} parameters")

# ============================================================================
# TRAINING
# ============================================================================

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print("\n" + "="*70)
print("TRAINING")
print("="*70)

num_epochs = 50
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss = 0.0
    
    for batch in train_loader:
        semantic_fp = batch['semantic_fingerprint'].to(device)
        motion_emb = batch['motion_type_embedding'].to(device)
        node_feat = batch['node_features'].to(device)
        adjacency = batch['adjacency'].to(device)
        target = batch['target_actions'].to(device)
        
        optimizer.zero_grad()
        
        # Use mean DOF for forward pass
        dof = int(batch['dof'].float().mean().item())
        
        output = model(semantic_fp, motion_emb, node_feat, adjacency, dof)
        
        # Loss on actual DOF
        loss = criterion(output[:, :dof], target[:, :dof])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            semantic_fp = batch['semantic_fingerprint'].to(device)
            motion_emb = batch['motion_type_embedding'].to(device)
            node_feat = batch['node_features'].to(device)
            adjacency = batch['adjacency'].to(device)
            target = batch['target_actions'].to(device)
            dof = int(batch['dof'].float().mean().item())
            
            output = model(semantic_fp, motion_emb, node_feat, adjacency, dof)
            loss = criterion(output[:, :dof], target[:, :dof])
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # LR scheduling
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    
    # Print
    print(f"Epoch {epoch+1:2d}/{num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}", end='')
    if new_lr != old_lr:
        print(f" | LR: {new_lr:.2e}")
    else:
        print()
    
    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '/scratch/anshb3/ovla/models/universal_mapper_best.pt')
        print(f"  ✓ Saved best model")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

print(f"\n✅ Best validation loss: {best_val_loss:.6f}")
print(f"✓ Saved to: /scratch/anshb3/ovla/models/universal_mapper_best.pt")
