"""
VLA Adapter Layer - Standardized interface for ANY VLA model

This is the ONLY place where VLA-specific knowledge lives.
The Universal Action Translator remains completely VLA-agnostic.
"""
from typing import Dict, Any, Tuple
import numpy as np


class VLAAdapter:
    """
    Base adapter class providing standardized interface for VLA models.
    
    Key principle: Separate VLA-specific knowledge (here) from 
    universal translation logic (UniversalActionTranslator).
    """
    
    def __init__(self, model, action_format: str, action_dim: int, 
                 horizon: int = 1, description: str = ""):
        """
        Args:
            model: The VLA model instance
            action_format: One of ['joint_position', 'end_effector_pose', 
                          'joint_velocity', 'cartesian_delta']
            action_dim: Dimensionality of action space
            horizon: Number of future timesteps predicted (1 for single action)
            description: Human-readable description of action space
        """
        self.model = model
        self.action_format = action_format
        self.action_dim = action_dim
        self.horizon = horizon
        self.description = description
        
    def predict(self, observation: Any, task: Any, **kwargs) -> np.ndarray:
        """
        Get actions from VLA model.
        
        Returns:
            actions: numpy array of shape (horizon, action_dim)
        """
        raise NotImplementedError("Subclass must implement predict()")
    
    def get_action_spec(self) -> Dict[str, Any]:
        """
        Get standardized action space specification.
        
        Returns:
            spec: Dictionary with format, dim, horizon, description
        """
        return {
            'format': self.action_format,
            'dim': self.action_dim,
            'horizon': self.horizon,
            'description': self.description
        }
    
    def __repr__(self):
        return (f"VLAAdapter(format={self.action_format}, "
                f"dim={self.action_dim}, horizon={self.horizon})")


class OpenVLAAdapter(VLAAdapter):
    """Adapter for OpenVLA model"""
    
    def __init__(self, model, processor):
        super().__init__(
            model=model,
            action_format='joint_position',
            action_dim=7,
            horizon=1,
            description='Normalized joint positions [j1, j2, j3, j4, j5, j6, gripper]'
        )
        self.processor = processor
        
    def predict(self, image, prompt: str, unnorm_key: str, **kwargs) -> np.ndarray:
        """
        Get action from OpenVLA.
        
        Args:
            image: PIL Image
            prompt: Text prompt for the task
            unnorm_key: Dataset key for unnormalization
            
        Returns:
            action: (1, 7) array of joint positions
        """
        # Import torch only when needed
        import torch
        
        inputs = self.processor(prompt, image).to(
            self.model.device, 
            dtype=torch.bfloat16
        )
        
        with torch.no_grad():
            action = self.model.predict_action(
                **inputs, 
                unnorm_key=unnorm_key, 
                do_sample=False
            )
        
        # OpenVLA returns (7,), we reshape to (1, 7) for consistency
        return action.flatten()[None, :]


class OctoAdapter(VLAAdapter):
    """Adapter for Octo model"""
    
    def __init__(self, model):
        super().__init__(
            model=model,
            action_format='cartesian_delta',
            action_dim=7,
            horizon=4,
            description='End-effector deltas [dx, dy, dz, droll, dpitch, dyaw, gripper]'
        )
        
    def predict(self, observations: Dict, task_text: str, rng, **kwargs) -> np.ndarray:
        """
        Get actions from Octo.
        
        Args:
            observations: Dict with 'image_primary', 'timestep_pad_mask', 'pad_mask_dict'
            task_text: Natural language task description
            rng: JAX random key
            
        Returns:
            actions: (4, 7) array of Cartesian deltas
        """
        # Create task dict
        task = self.model.create_tasks(texts=[task_text])
        task_lang_only = {
            'language_instruction': task['language_instruction'],
            'pad_mask_dict': {
                'language_instruction': task['pad_mask_dict']['language_instruction']
            }
        }
        
        # Sample actions
        actions = self.model.sample_actions(observations, task_lang_only, rng=rng)
        
        # Octo returns (1, 4, 7), we return (4, 7)
        return actions[0]


class RT2Adapter(VLAAdapter):
    """Adapter for RT-2 model (placeholder - needs real implementation)"""
    
    def __init__(self, model):
        super().__init__(
            model=model,
            action_format='end_effector_pose',  # TODO: Verify this
            action_dim=7,
            horizon=1,
            description='End-effector pose [x, y, z, qx, qy, qz, qw] (TODO: verify)'
        )
        
    def predict(self, observation: Any, task: Any, **kwargs) -> np.ndarray:
        """TODO: Implement RT-2 prediction"""
        raise NotImplementedError("RT-2 adapter not yet implemented")
