import cv2
import numpy as np
import os

print("Working directory:", os.getcwd())
print("Script directory:", os.path.dirname(__file__))

# Get folder where script lives
base_dir = os.path.dirname(__file__)

# Build full path to image
image_path = os.path.join(base_dir, "input.png")

print("Trying to load:", image_path)
    
# Load image
img = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray, 50, 150)

# Invert edges
edges = cv2.bitwise_not(edges)

# Create blueprint background
blue = np.zeros_like(img)
blue[:] = (128, 0, 0)  # dark blue in BGR

# Convert edges to 3-channel
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Combine
result = cv2.bitwise_and(edges_colored, blue)

cv2.imwrite("blueprint.jpg", result)