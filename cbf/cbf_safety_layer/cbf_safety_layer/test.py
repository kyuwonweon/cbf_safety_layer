import numpy as np

def closest_points_segments(p1, q1, p2, q2):
    """
    Find the closest points between Segment A (p1->q1) and Segment B (p2->q2).
    
    Args:
        p1, q1: Start and End points of Segment A (numpy arrays)
        p2, q2: Start and End points of Segment B (numpy arrays)
        
    Returns:
        c1: Closest point on Segment A
        c2: Closest point on Segment B
        dist: The distance between them
    """
    
    # Direction vectors
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    
    # Squared length of segments
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    # Check if either segment is just a point (length ~ 0)
    if a <= 1e-6 and e <= 1e-6:
        # Both are points
        s, t = 0.0, 0.0
    elif a <= 1e-6:
        # Segment A is a point
        s = 0.0
        t = f / e
        t = np.clip(t, 0.0, 1.0)
    elif e <= 1e-6:
        # Segment B is a point
        t = 0.0
        s = -np.dot(d1, r) / a
        s = np.clip(s, 0.0, 1.0)
    else:
        # General case (Standard line-line distance)
        c = np.dot(d1, r)
        b = np.dot(d1, d2)
        denom = a * e - b * b # Determinant
        
        # If segments are not parallel
        if denom != 0.0:
            s = (b * f - c * e) / denom
        else:
            # Parallel lines
            s = 0.0
            
        # Compute t based on s
        t = (b * s + f) / e
        
        # --- CLAMPING LOGIC (The "Segment" part) ---
        # If s went out of bounds [0, 1], clamp it and re-solve for t
        if s < 0.0:
            s = 0.0
            t = f / e
            t = np.clip(t, 0.0, 1.0)
        elif s > 1.0:
            s = 1.0
            t = (b + f) / e
            t = np.clip(t, 0.0, 1.0)
            
        # If t went out of bounds, clamp it and re-solve for s
        # (We only need to re-solve for s if t changed significantly)
        if t < 0.0:
            t = 0.0
            s = -c / a
            s = np.clip(s, 0.0, 1.0)
        elif t > 1.0:
            t = 1.0
            s = (b - c) / a
            s = np.clip(s, 0.0, 1.0)
            
    # Calculate the actual points
    c1 = p1 + d1 * s
    c2 = p2 + d2 * t
    
    dist = np.linalg.norm(c1 - c2)
    
    return c1, c2, dist

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Test Case: Two lines crossing at different heights (like an X)
    # Segment A: Along X-axis
    p1 = np.array([-1.0, 0.0, 0.0])
    q1 = np.array([1.0, 0.0, 0.0])
    
    # Segment B: Along Y-axis, but lifted up by 0.5
    p2 = np.array([0.0, -1.0, 0.0])
    q2 = np.array([0.0, 1.0, 0.5])
    
    c1, c2, d = closest_points_segments(p1, q1, p2, q2)
    
    print(f"Distance: {d}") 
    print(f"Point on A: {c1}")
    print(f"Point on B: {c2}")
    
    # Expected: Distance should be 0.5 (the vertical gap)