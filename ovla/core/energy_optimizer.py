import numpy as np

class EnergyOptimizer:
    """
    Layer 3 Strategy: Energy Minimization.
    Universally reads URDF effort limits and applies proportional dampening 
    to heavily-loaded joints, forcing lighter joints to carry the movement intent.
    """
    def __init__(self, robot_profile):
        self.profile = robot_profile
        joints = self.profile.get('joints', [])
        efforts = np.array([j.get('max_effort', 1.0) for j in joints])
        
        if len(efforts) > 0 and np.max(efforts) > 0:
            self.weights = efforts / np.max(efforts)
        else:
            self.weights = np.ones(6)

    def optimize_target(self, current_joints, vla_target_joints):
        current_joints = np.array(current_joints)
        vla_target_joints = np.array(vla_target_joints)
        
        # Raw VLA intent
        delta = vla_target_joints - current_joints
        
        # Exponential proportional dampening based on URDF weights.
        # A heavy base joint (weight 1.0) might only execute 74% of its commanded delta.
        # A light wrist joint (weight 0.1) will execute 90%+ of its commanded delta.
        dampening_factors = np.exp(-0.3 * self.weights)
        
        optimized_delta = delta * dampening_factors
        optimized_target = current_joints + optimized_delta
        
        # Calculate theoretical power draw (Effort * absolute movement)
        original_energy = np.sum(self.weights * np.abs(delta))
        new_energy = np.sum(self.weights * np.abs(optimized_delta))
        saved_pct = ((original_energy - new_energy) / original_energy) * 100 if original_energy > 0 else 0
        
        return optimized_target, saved_pct
