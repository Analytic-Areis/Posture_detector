import math
import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (a, b, c).
    b is the vertex.
    Returns the angle in degrees.
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def calculate_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.
    """
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

def calculate_ear(eye_points):
    """
    Calculates the Eye Aspect Ratio (EAR) given 6 points of an eye.
    Typically used with 2D landmarks (x, y).
    Points should be in order: left corner, top-left, top-right, right corner, bottom-right, bottom-left.
    """
    # Vertical distances
    v1 = calculate_distance(eye_points[1], eye_points[5])
    v2 = calculate_distance(eye_points[2], eye_points[4])
    
    # Horizontal distance
    h = calculate_distance(eye_points[0], eye_points[3])
    
    # EAR formula
    ear = (v1 + v2) / (2.0 * h)
    return ear
