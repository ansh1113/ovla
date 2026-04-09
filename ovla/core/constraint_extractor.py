import torch
import torch.nn as nn

class GeometricConstraintNet(nn.Module):
    """
    Layer 2: Extracts physical 3D constraints from visual embeddings.
    Multi-head architecture predicting spatial boundaries without human labels.
    """
    def __init__(self, embedding_dim=128):
        super().__init__()
        
        # Shared visual backbone
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Head 1: 3D Center (x, y, z relative to robot base)
        self.center_head = nn.Linear(64, 3)
        
        # Head 2: Tolerance Radius (absolute distance in meters)
        self.radius_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Softplus() # Ensures radius is always strictly positive
        )
        
        # Head 3: Orientation Bounds (Min/Max for Roll, Pitch, Yaw -> 6 values)
        self.orientation_head = nn.Linear(64, 6)
        
    def forward(self, x):
        shared_features = self.shared(x)
        
        center = self.center_head(shared_features)
        radius = self.radius_head(shared_features)
        orientation_bounds = self.orientation_head(shared_features)
        
        return {
            'center_xyz': center,
            'tolerance_r': radius,
            'orientation_bounds': orientation_bounds
        }
