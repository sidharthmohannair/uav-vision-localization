import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to detect keypoints and compute descriptors
def detect_and_compute(detector_name, image):
    if detector_name == "sift":
        detector = cv2.SIFT_create()
    elif detector_name == "orb":
        detector = cv2.ORB_create()
    else:
        raise ValueError(f"Unsupported detector: {detector_name}")
    
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors

# Function to match features
def match_features(desc1, desc2, detector_name):
    if detector_name == "sift":  # Floating-point descriptors
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif detector_name == "orb":  # Binary descriptors
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        raise ValueError(f"Unsupported detector: {detector_name}")
    
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# Function to find the drone's location relative to the orthomosaic
def find_location(orthomosaic, drone_image, detector_name):
    kp1, desc1 = detect_and_compute(detector_name, orthomosaic)
    kp2, desc2 = detect_and_compute(detector_name, drone_image)
    
    # Debug descriptor types
    print(f"Orthomosaic descriptor type: {desc1.dtype}, shape: {desc1.shape}")
    print(f"Drone image descriptor type: {desc2.dtype}, shape: {desc2.shape}")
    
    matches = match_features(desc1, desc2, detector_name)
    
    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography.")
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    h, w = drone_image.shape[:2]
    drone_center = np.array([[w / 2, h / 2]], dtype="float32").reshape(-1, 1, 2)
    satellite_position = cv2.perspectiveTransform(drone_center, H)
    
    return satellite_position[0][0], matches, kp1, kp2

# Function to visualize matches
def visualize_matches(orthomosaic, drone_image, kp1, kp2, matches, detector_name):
    matched_image = cv2.drawMatches(
        orthomosaic, kp1, drone_image, kp2, matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0),  # Green color for matches
        singlePointColor=(255, 0, 0)  # Blue color for keypoints
    )
    plt.figure(figsize=(15, 10))
    plt.title(f"Feature Matches ({detector_name.upper()})", fontsize=16)
    plt.imshow(matched_image, cmap="gray")
    plt.axis("off")
    plt.show()

# Main function
if __name__ == "__main__":
    # Load images
    orthomosaic = cv2.imread("satellite_image1.jpg", cv2.IMREAD_GRAYSCALE)
    drone_image = cv2.imread("crop_image.jpg", cv2.IMREAD_GRAYSCALE)
    
    if orthomosaic is None or drone_image is None:
        raise ValueError("Failed to load one or both images.")
    
    # Choose detector (e.g., "sift" or "orb")
    detector_name = "sift"
    
    # Find drone's location
    try:
        position, matches, kp1, kp2 = find_location(orthomosaic, drone_image, detector_name)
        print(f"Drone's estimated position in orthomosaic: {position}")
        
        # Visualize matches
        visualize_matches(orthomosaic, drone_image, kp1, kp2, matches, detector_name)
    except Exception as e:
        print(f"Error: {str(e)}")
