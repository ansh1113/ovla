import numpy as np

class WorkspaceOptimizer:
    """
    Milestone 2.4: Mathematically shortens inefficient human teleoperation paths.
    Uses 3D Line-Sphere Intersection to guarantee the shortened path does not 
    collide with geometric constraints (keep-out zones).
    """
    def __init__(self):
        pass

    def calculate_path_length(self, waypoints):
        length = 0.0
        for i in range(1, len(waypoints)):
            length += np.linalg.norm(waypoints[i] - waypoints[i-1])
        return length

    def _line_sphere_intersect(self, p1, p2, sphere_center, sphere_radius):
        """
        Calculates if a 3D line segment intersects a 3D sphere.
        Uses pure vector math: quadratic equation of the line parameterized by t.
        """
        d = p2 - p1
        f = p1 - sphere_center
        
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - sphere_radius**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return False # No intersection with the infinite line
            
        # Check if the intersection happens within the segment bounds (t between 0 and 1)
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a + 1e-8)
        t2 = (-b + discriminant) / (2*a + 1e-8)
        
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True # The segment physically hits the sphere
            
        return False

    def optimize_path(self, demo_waypoints, constraint_spheres=[]):
        """
        Attempts to straighten the path. Reverts to safe demo if obstacle detected.
        constraint_spheres: List of dicts [{'center_xyz': array, 'tolerance_r': float}]
        """
        if len(demo_waypoints) < 2:
            return demo_waypoints, 0.0, 0.0, 0.0
            
        start_pt = demo_waypoints[0]
        goal_pt = demo_waypoints[-1]
        demo_length = self.calculate_path_length(demo_waypoints)
        
        # 1. Propose the mathematically perfect straight line
        direct_length = np.linalg.norm(goal_pt - start_pt)
        
        # 2. Collision Detection: Ray-cast the straight line against all constraint spheres
        path_is_clear = True
        for sphere in constraint_spheres:
            if self._line_sphere_intersect(start_pt, goal_pt, sphere['center_xyz'], sphere['tolerance_r']):
                path_is_clear = False
                break
                
        # 3. Decision Logic
        if path_is_clear:
            # Safe to optimize! Generate the direct vector.
            optimized_waypoints = np.linspace(start_pt, goal_pt, num=len(demo_waypoints))
            reduction = (demo_length - direct_length) / demo_length
            return optimized_waypoints, demo_length, direct_length, reduction
        else:
            # Obstacle in the way! Fall back to the safe, human-demonstrated curve.
            return demo_waypoints, demo_length, demo_length, 0.0
