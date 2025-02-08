import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class FeatureMatcher:
    detector_name: str
    
    def detect_and_compute(self, image):
        """Detect keypoints and compute descriptors using the specified detector."""
        detectors = {
            "sift": cv2.SIFT_create(),
            "orb": cv2.ORB_create(),
            "akaze": cv2.AKAZE_create(),
            "brisk": cv2.BRISK_create()
        }
        
        if self.detector_name not in detectors:
            raise ValueError(f"Unsupported detector: {self.detector_name}")
        
        detector = detectors[self.detector_name]
        keypoints, descriptors = detector.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """Match features using FLANN or BFMatcher."""
        if self.detector_name == "sift":
            # Use FLANN for floating-point descriptors (SIFT)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # Use BFMatcher for binary descriptors (ORB, AKAZE, BRISK)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        matches = matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    
    def find_location(self, satellite_image, drone_image):
        """Find the drone's location relative to the satellite image."""
        kp1, desc1 = self.detect_and_compute(satellite_image)
        kp2, desc2 = self.detect_and_compute(drone_image)
        
        if desc1 is None or desc2 is None:
            raise ValueError("No descriptors found in one or both images.")
        
        matches = self.match_features(desc1, desc2)
        
        if len(matches) < 4:
            raise ValueError("Not enough matches to compute homography.")
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        h, w = drone_image.shape[:2]
        drone_center = np.array([[w / 2, h / 2]], dtype="float32").reshape(-1, 1, 2)
        satellite_position = cv2.perspectiveTransform(drone_center, H)
        
        return satellite_position[0][0], matches, kp1, kp2


@dataclass
class HybridFeatureMatcher:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.sift = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.flann_matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=50)
        )
    
    def detect_and_compute(self, image):
        """Detect keypoints and compute descriptors using ORB and SIFT."""
        orb_kp, orb_desc = self.orb.detectAndCompute(image, None)
        sift_kp, sift_desc = self.sift.detectAndCompute(image, None)
        return orb_kp, orb_desc, sift_kp, sift_desc
    
    def match_features(self, desc1, desc2):
        """Match features using a combination of ORB and SIFT."""
        orb_matches = []
        sift_matches = []
        
        # Match ORB features
        if desc1[1] is not None and desc2[1] is not None:
            orb_matches = self.bf_matcher.match(desc1[1], desc2[1])
            orb_matches = sorted(orb_matches, key=lambda x: x.distance)[:100]
        
        # Match SIFT features
        if desc1[3] is not None and desc2[3] is not None:
            sift_matches = self.flann_matcher.match(desc1[3], desc2[3])
            sift_matches = sorted(sift_matches, key=lambda x: x.distance)[:100]
        
        combined_matches = orb_matches + sift_matches
        return combined_matches
    
    def refine_homography(self, src_pts, dst_pts):
        """Refine homography using Levenberg-Marquardt optimization."""
        def residuals(H, src_pts, dst_pts):
            H = H.reshape(3, 3)
            projected = cv2.perspectiveTransform(src_pts, H)
            return (projected - dst_pts).ravel()
        
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        result = least_squares(residuals, H.ravel(), args=(src_pts, dst_pts))
        return result.x.reshape(3, 3)
    
    def find_location(self, satellite_image, drone_image):
        """Find the drone's location relative to the satellite image."""
        orb_kp1, orb_desc1, sift_kp1, sift_desc1 = self.detect_and_compute(satellite_image)
        orb_kp2, orb_desc2, sift_kp2, sift_desc2 = self.detect_and_compute(drone_image)
        
        if (orb_desc1 is None and sift_desc1 is None) or (orb_desc2 is None and sift_desc2 is None):
            raise ValueError("No descriptors found in one or both images.")
        
        matches = self.match_features(
            (orb_kp1, orb_desc1, sift_kp1, sift_desc1),
            (orb_kp2, orb_desc2, sift_kp2, sift_desc2)
        )
        
        if len(matches) < 4:
            raise ValueError("Not enough matches to compute homography.")
        
        src_pts = np.float32([orb_kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([orb_kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H = self.refine_homography(dst_pts, src_pts)
        
        h, w = drone_image.shape[:2]
        drone_center = np.array([[w / 2, h / 2]], dtype="float32").reshape(-1, 1, 2)
        satellite_position = cv2.perspectiveTransform(drone_center, H)
        
        return satellite_position[0][0], matches, orb_kp1, orb_kp2


@dataclass
class TestResult:
    detector_name: str
    computation_time: float
    matches_count: int
    position_error: float
    homography_accuracy: float


class DroneLocalizationEvaluator:
    def __init__(self, satellite_image_path: str, drone_image_path: str):
        self.satellite_image = cv2.imread(satellite_image_path, cv2.IMREAD_GRAYSCALE)
        self.drone_image = cv2.imread(drone_image_path, cv2.IMREAD_GRAYSCALE)
        
        if self.satellite_image is None or self.drone_image is None:
            raise ValueError("Failed to load one or both images.")
        
        self.detectors = ["sift", "orb", "akaze", "brisk", "hfm"]
        self.results = []
    
    def evaluate_detectors(self):
        """Evaluate all detectors and compute metrics."""
        for detector in self.detectors:
            try:
                start_time = time.time()
                
                if detector == "hfm":
                    finder = HybridFeatureMatcher()
                else:
                    finder = FeatureMatcher(detector)
                
                position, matches, kp1, kp2 = finder.find_location(self.satellite_image, self.drone_image)
                
                computation_time = time.time() - start_time
                matches_count = len(matches)
                
                # Calculate position error
                true_position = np.array([self.satellite_image.shape[1] / 2, self.satellite_image.shape[0] / 2])
                position_error = np.linalg.norm(position - true_position)
                
                # Calculate homography accuracy
                homography_accuracy = matches_count / (len(kp1) + len(kp2))
                
                self.results.append(TestResult(
                    detector_name=detector,
                    computation_time=computation_time,
                    matches_count=matches_count,
                    position_error=position_error,
                    homography_accuracy=homography_accuracy
                ))
                
                self.visualize_matches(detector, kp1, kp2, matches)
                self.visualize_drone_position(position)
            
            except Exception as e:
                logging.error(f"Error evaluating {detector}: {str(e)}")
    
    def visualize_matches(self, detector: str, kp1, kp2, matches):
        """Visualize feature matches."""
        matched_image = cv2.drawMatches(
            self.satellite_image, kp1, self.drone_image, kp2, matches[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0),  # Green color for matches
            singlePointColor=(255, 0, 0)  # Blue color for keypoints
        )
        
        plt.figure(figsize=(15, 10))
        plt.title(f"Feature Matches ({detector.upper()})", fontsize=16)
        plt.imshow(matched_image, cmap="gray")
        plt.axis("off")
        plt.savefig(f"matches_{detector}.png", bbox_inches="tight", dpi=300)
        plt.close()
    
    def visualize_drone_position(self, position):
        """Visualize the drone's position on the satellite image."""
        satellite_color = cv2.cvtColor(self.satellite_image, cv2.COLOR_GRAY2BGR)
        cv2.circle(satellite_color, (int(position[0]), int(position[1])), radius=10, color=(0, 0, 255), thickness=-1)
        cv2.putText(
            satellite_color, "Drone Position", 
            (int(position[0]) + 15, int(position[1])), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        
        plt.figure(figsize=(15, 10))
        plt.title("Drone's Estimated Position", fontsize=16)
        plt.imshow(satellite_color, cmap="gray")
        plt.axis("off")
        plt.savefig("drone_position.png", bbox_inches="tight", dpi=300)
        plt.close()
    
    def analyze_results(self):
        """Analyze and summarize the results."""
        df = pd.DataFrame([vars(r) for r in self.results])
        print("\nDetector Performance Summary:")
        print(df)
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Detector Performance Comparison", fontsize=18)
        
        df.plot(kind="bar", x="detector_name", y="computation_time", ax=axes[0, 0], title="Computation Time (s)", color="skyblue")
        df.plot(kind="bar", x="detector_name", y="matches_count", ax=axes[0, 1], title="Number of Matches", color="lightgreen")
        df.plot(kind="bar", x="detector_name", y="position_error", ax=axes[1, 0], title="Position Error (pixels)", color="salmon")
        df.plot(kind="bar", x="detector_name", y="homography_accuracy", ax=axes[1, 1], title="Homography Accuracy", color="gold")
        
        plt.tight_layout()
        plt.savefig("performance_summary.png", bbox_inches="tight", dpi=300)
        plt.close()


# Example usage
if __name__ == "__main__":
    evaluator = DroneLocalizationEvaluator(
        satellite_image_path="satellite_image.jpg",
        drone_image_path="drone_image3.jpg"
    )
    evaluator.evaluate_detectors()
    evaluator.analyze_results()