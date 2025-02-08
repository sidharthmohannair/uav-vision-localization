import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class TestCase:
    name: str
    satellite_image: np.ndarray
    drone_image: np.ndarray
    description: str

class FeatureMatcher:
    def __init__(self, detector_name):
        self.detector_name = detector_name
        self.detector = self._create_detector()

    def _create_detector(self):
        if self.detector_name == "sift":
            return cv2.SIFT_create(nfeatures=2000)  # Increased number of features
        elif self.detector_name == "orb":
            return cv2.ORB_create(nfeatures=2000)
        elif self.detector_name == "akaze":
            return cv2.AKAZE_create()
        elif self.detector_name == "brisk":
            return cv2.BRISK_create()
        else:
            raise ValueError(f"Unknown detector: {self.detector_name}")

    def find_location(self, satellite_image, drone_image):
        try:
            # Detect keypoints and descriptors
            kp1, des1 = self.detector.detectAndCompute(satellite_image, None)
            kp2, des2 = self.detector.detectAndCompute(drone_image, None)

            if kp1 is None or kp2 is None or len(kp1) == 0 or len(kp2) == 0:
                raise ValueError(f"No keypoints found")

            # Match features with more lenient parameters
            if self.detector_name == "sift":
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=100)  # Increased checks
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
                matches = matcher.knnMatch(des1, des2, k=2)
                
                # Apply ratio test with more lenient threshold
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.8 * n.distance:  # Increased ratio threshold
                        good_matches.append(m)
                matches = good_matches
            else:
                # For binary descriptors
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(des1, des2)
                # Sort matches by distance
                matches = sorted(matches, key=lambda x: x.distance)
                # Take top 80% of matches
                matches = matches[:int(len(matches) * 0.8)]

            if len(matches) < 4:
                raise ValueError(f"Not enough matches found: {len(matches)}")

            # Extract matched keypoints
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Find homography with more lenient parameters
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0, maxIters=2000)
            if H is None:
                raise ValueError("Could not compute homography")

            # Calculate center position
            h, w = drone_image.shape[:2]
            drone_center = np.array([[w/2, h/2]], dtype=np.float32).reshape(-1, 1, 2)
            transformed_center = cv2.perspectiveTransform(drone_center, H)

            return transformed_center[0][0], matches, kp1, kp2

        except Exception as e:
            logging.error(f"Error in feature matching: {str(e)}")
            raise

class PracticalDroneEvaluator:
    def __init__(self, satellite_path: str, drone_images: List[str]):
        """
        Initialize evaluator with satellite image and list of drone image paths
        """
        self.satellite_image = cv2.imread(satellite_path, cv2.IMREAD_GRAYSCALE)
        if self.satellite_image is None:
            raise ValueError(f"Could not load satellite image from {satellite_path}")
        
        self.drone_images = drone_images    
        self.detectors = ["sift", "orb", "akaze", "brisk"]
        self.test_cases = self.prepare_test_cases()
        self.results = []

    def prepare_test_cases(self) -> List[TestCase]:
        """
        Prepare test cases from drone images
        """
        test_cases = []
        
        # Define test scenarios
        test_scenarios = {
            "normal": {
                "description": "Standard view comparison",
                "transform": None
            },
            "rotated": {
                "description": "Drone image rotated 45 degrees",
                "transform": lambda img: self.rotate_image(img, 45)
            },
            "scaled": {
                "description": "Drone image scaled to 80%",
                "transform": lambda img: cv2.resize(img, None, fx=0.8, fy=0.8)
            },
            "brightness": {
                "description": "Increased brightness",
                "transform": lambda img: cv2.convertScaleAbs(img, alpha=1.5, beta=0)
            },
            "noise": {
                "description": "Added Gaussian noise",
                "transform": self.add_noise
            }
        }

        # Process each drone image
        for image_path in self.drone_images:
            drone_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if drone_img is None:
                logging.warning(f"Could not load image {image_path}")
                continue

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Create test cases for each scenario
            for scenario, config in test_scenarios.items():
                test_img = config["transform"](drone_img.copy()) if config["transform"] else drone_img
                test_name = f"{base_name}_{scenario}"
                test_cases.append(TestCase(
                    name=test_name,
                    satellite_image=self.satellite_image,
                    drone_image=test_img,
                    description=f"{base_name}: {config['description']}"
                ))

        return test_cases

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        height, width = image.shape[:2]
        center = (width/2, height/2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        noisy_img = cv2.add(image, noise)
        return noisy_img

    def evaluate_detector(self, detector_name: str, test_case: TestCase) -> Dict:
        """
        Evaluate a single detector on a test case
        """
        try:
            start_time = time.time()
            
            # Create detector instance
            finder = FeatureMatcher(detector_name)
            
            # Find location
            position, matches, kp1, kp2 = finder.find_location(
                test_case.satellite_image, 
                test_case.drone_image
            )
            
            computation_time = time.time() - start_time
            matches_count = len(matches)
            match_quality = len([m for m in matches if m.distance < 50]) / len(matches) if matches else 0

            result = {
                'detector': detector_name,
                'test_case': test_case.name,
                'description': test_case.description,
                'computation_time': computation_time,
                'matches_count': matches_count,
                'match_quality': match_quality,
                'position_x': position[0],
                'position_y': position[1],
                'success': True,
                'error': None
            }

            # Save visualization
            self.save_visualization(test_case, detector_name, position, matches, kp1, kp2)
            return result

        except Exception as e:
            logging.error(f"Error in {detector_name} for {test_case.name}: {str(e)}")
            return {
                'detector': detector_name,
                'test_case': test_case.name,
                'description': test_case.description,
                'computation_time': 0,
                'matches_count': 0,
                'match_quality': 0,
                'position_x': None,
                'position_y': None,
                'success': False,
                'error': str(e)
            }

    def save_visualization(self, test_case: TestCase, detector: str, 
                         position: Tuple[float, float], matches, kp1, kp2):
        """Save visualization of matches and detected position"""
        try:
            # Create output directory
            os.makedirs(f"results/{test_case.name}", exist_ok=True)

            # Draw matches
            if len(matches) > 0:
                matched_img = cv2.drawMatches(
                    test_case.satellite_image, kp1,
                    test_case.drone_image, kp2,
                    matches[:50], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                plt.figure(figsize=(15, 8))
                plt.title(f"Matches for {detector} - {test_case.name}\n"
                         f"({len(matches)} matches found)")
                plt.imshow(matched_img, cmap='gray')
                plt.axis('off')
                plt.savefig(f"results/{test_case.name}/matches_{detector}.png")
                plt.close()

            # Draw position
            result_img = cv2.cvtColor(test_case.satellite_image, cv2.COLOR_GRAY2BGR)
            cv2.circle(result_img, (int(position[0]), int(position[1])), 
                      10, (0, 0, 255), -1)
            
            plt.figure(figsize=(10, 10))
            plt.title(f"Detected Position - {detector} - {test_case.name}")
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.savefig(f"results/{test_case.name}/position_{detector}.png")
            plt.close()

        except Exception as e:
            logging.error(f"Error saving visualization for {detector} - {test_case.name}: {str(e)}")

    def run_evaluation(self):
        """
        Run evaluation for all detectors and test cases
        """
        results = []
        
        for test_case in self.test_cases:
            print(f"\nEvaluating test case: {test_case.name}")
            for detector in self.detectors:
                print(f"  Testing detector: {detector}")
                result = self.evaluate_detector(detector, test_case)
                results.append(result)

        self.results = pd.DataFrame(results)
        return self.results

    def generate_report(self):
        """
        Generate comprehensive report of results
        """
        if len(self.results) == 0:
            raise ValueError("No results available. Run evaluation first.")

        # Create report directory
        os.makedirs("report", exist_ok=True)

        # Overall performance by detector
        detector_stats = self.results.groupby('detector').agg({
            'computation_time': ['mean', 'std'],
            'matches_count': ['mean', 'std'],
            'match_quality': ['mean', 'std'],
            'success': 'mean'
        }).round(3)

        # Performance by test case
        test_case_stats = self.results.groupby(['test_case', 'description', 'detector']).agg({
            'computation_time': 'mean',
            'matches_count': 'mean',
            'match_quality': 'mean',
            'success': 'mean'
        }).round(3)

        # Save statistics
        detector_stats.to_csv("report/detector_statistics.csv")
        test_case_stats.to_csv("report/test_case_statistics.csv")

        # Create visualization of results
        self.visualize_results()

        # Generate summary report
        with open("report/summary.txt", "w") as f:
            f.write("=== Drone Localization Evaluation Summary ===\n\n")
            
            f.write("Test Cases Evaluated:\n")
            for test_case in self.test_cases:
                f.write(f"- {test_case.name}: {test_case.description}\n")
            
            f.write("\nDetector Performance Summary:\n")
            f.write(detector_stats.to_string())
            
            f.write("\n\nBest Performer by Category:\n")
            f.write(f"Fastest: {self.results.groupby('detector')['computation_time'].mean().idxmin()}\n")
            f.write(f"Most Matches: {self.results.groupby('detector')['matches_count'].mean().idxmax()}\n")
            f.write(f"Best Quality: {self.results.groupby('detector')['match_quality'].mean().idxmax()}\n")
            f.write(f"Most Reliable: {self.results.groupby('detector')['success'].mean().idxmax()}\n")

    def visualize_results(self):
        """
        Create visualizations of results
        """
        # Use default style instead of seaborn
        plt.style.use('default')
        
        try:
            # Performance comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("Detector Performance Comparison")

            # Computation time
            self.results.boxplot(column='computation_time', by='detector', ax=axes[0,0])
            axes[0,0].set_title("Computation Time")
            axes[0,0].set_ylabel("Seconds")

            # Match count
            self.results.boxplot(column='matches_count', by='detector', ax=axes[0,1])
            axes[0,1].set_title("Number of Matches")

            # Match quality
            self.results.boxplot(column='match_quality', by='detector', ax=axes[1,0])
            axes[1,0].set_title("Match Quality")

            # Success rate
            success_rate = self.results.groupby('detector')['success'].mean()
            success_rate.plot(kind='bar', ax=axes[1,1])
            axes[1,1].set_title("Success Rate")
            axes[1,1].set_ylabel("Rate")

            plt.tight_layout()
            plt.savefig("report/performance_comparison.png")
            plt.close()
        except Exception as e:
            logging.error(f"Error in visualization: {str(e)}")
            # Continue execution even if visualization fails

# Example usage
if __name__ == "__main__":
    try:
        # Create output directories
        os.makedirs("results", exist_ok=True)
        os.makedirs("report", exist_ok=True)

        # Initialize evaluator with your images
        drone_images = [
            "drone_image.jpg",
            "drone_image2.jpg",
            "drone_image3.jpg"
        ]
        
        evaluator = PracticalDroneEvaluator(
            satellite_path="satellite_image.jpg",
            drone_images=drone_images
        )
        
        # Run evaluation
        results = evaluator.run_evaluation()
        
        # Generate report
        evaluator.generate_report()
        
        print("Evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
