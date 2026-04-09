import numpy as np
from scipy.interpolate import CubicSpline

class TrajectoryGenerator:
    """
    Layer 4: Trajectory Generation.
    Converts discrete mathematical targets into safe, continuous 50Hz motor commands.
    """
    def __init__(self, robot_profile, control_freq=50):
        self.profile = robot_profile
        self.hz = control_freq

    def generate_smooth_path(self, current_joints, target_joints, duration=2.0):
        current_joints = np.array(current_joints)
        target_joints = np.array(target_joints)
        
        # We need time boundaries
        times = [0.0, duration]
        waypoints = np.vstack((current_joints, target_joints))
        
        # 'clamped' enforces boundary conditions: velocity is exactly 0.0 at start and end
        cs = CubicSpline(times, waypoints, bc_type='clamped')
        
        # Mathematically sample the curve at 50Hz
        num_steps = max(2, int(duration * self.hz))
        sample_times = np.linspace(0, duration, num_steps)
        
        trajectory = cs(sample_times)
        return trajectory
