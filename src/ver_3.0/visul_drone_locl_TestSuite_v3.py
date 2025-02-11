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

class PositionError:
    def __init__(self):
        self.errors = []
        
    def calculate_error(self, estimated_pos, actual_pos):
        """
        Calculate Euclidean distance between estimated and actual positions
        """
        if estimated_pos is None or actual_pos is None:
            return None
            
        error = np.sqrt(
            (estimated_pos[0] - actual_pos[0])**2 + 
            (estimated_pos[1] - actual_pos[1])**2
        )
        return error
        
    def add_error(self, detector_name, test_case, estimated_pos, actual_pos):
        """
        Add error measurement to the collection
        """
        error = self.calculate_error(estimated_pos, actual_pos)
        self.errors.append({
            'detector': detector_name,
            'test_case': test_case,
            'error': error,
            'estimated_pos': estimated_pos,
            'actual_pos': actual_pos
        })
        
    def visualize_errors(self, satellite_image):
        """
        Visualize errors on the satellite image
        """
        # Create figure for error visualization
        plt.figure(figsize=(15, 10))
        
        # Convert grayscale to RGB for colored annotations
        if len(satellite_image.shape) == 2:
            vis_img = cv2.cvtColor(satellite_image, cv2.COLOR_GRAY2RGB)
        else:
            vis_img = satellite_image.copy()
            
        # Plot all positions and errors
        for error_data in self.errors:
            if error_data['estimated_pos'] is not None and error_data['actual_pos'] is not None:
                # Plot actual position (green)
                actual_x, actual_y = error_data['actual_pos']
                plt.plot(actual_x, actual_y, 'go', markersize=10, label='Actual Position')
                
                # Plot estimated position (red)
                est_x, est_y = error_data['estimated_pos']
                plt.plot(est_x, est_y, 'ro', markersize=10, label='Estimated Position')
                
                # Draw line between actual and estimated positions
                plt.plot([actual_x, est_x], [actual_y, est_y], 'y--', alpha=0.5)
                
                # Add error value text
                mid_x = (actual_x + est_x) / 2
                mid_y = (actual_y + est_y) / 2
                plt.text(mid_x, mid_y, f'Error: {error_data["error"]:.2f}px',
                        color='white', bbox=dict(facecolor='black', alpha=0.7))
        
        plt.imshow(vis_img)
        plt.title('Position Estimation Errors')
        plt.legend()
        plt.grid(True)
        plt.savefig('report/position_errors.png')
        plt.close()
        
    def generate_error_report(self):
        """
        Generate statistical report of position errors
        """
        error_df = pd.DataFrame(self.errors)
        
        # Calculate statistics per detector
        stats = error_df.groupby('detector').agg({
            'error': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        # Add success rate (percentage of non-None errors)
        success_rate = error_df.groupby('detector')['error'].apply(
            lambda x: (x.notna().sum() / len(x) * 100)
        ).round(2)
        stats['error', 'success_rate'] = success_rate
        
        return stats


@dataclass
class TestCase:
    name: str
    satellite_image: np.ndarray
    drone_image: np.ndarray
    description: str
    actual_position: Tuple[float, float]  # Add ground truth position

class FeatureMatcher:
    def __init__(self, detector_name):
        self.detector_name = detector_name
        self.detector = self._create_detector()

    # Add this new method here
    def preprocess_image(self, image):
        """
        Preprocess image to improve feature detection in noisy conditions
        """
        denoised = cv2.fastNlMeansDenoising(image, None, h=10, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        blurred = cv2.GaussianBlur(enhanced, (3,3), 0)
        return blurred

    # Add this new method here
    def estimate_noise(self, image):
        """
        Estimate noise level in image
        Returns a value between 0 and 1
        """
        mean, std = cv2.meanStdDev(image)
        noise_estimate = min(1.0, std[0][0] / 128.0)
        return noise_estimate

    # Replace your existing _create_detector method with this:
    def _create_detector(self):
        if self.detector_name == "sift":
            return cv2.SIFT_create(
                nfeatures=3000,
                nOctaveLayers=5,
                contrastThreshold=0.03,
                edgeThreshold=15
            )
        elif self.detector_name == "orb":
            return cv2.ORB_create(
                nfeatures=3000,
                scaleFactor=1.1,
                nlevels=12,
                edgeThreshold=31,
                firstLevel=0,
                WTA_K=3,
                patchSize=31
            )
        elif self.detector_name == "akaze":
            return cv2.AKAZE_create(
                descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                descriptor_size=0,
                descriptor_channels=3,
                threshold=0.001
            )
        elif self.detector_name == "brisk":
            return cv2.BRISK_create(
                thresh=30,
                octaves=4,
                patternScale=1.0
            )
        else:
            raise ValueError(f"Unknown detector: {self.detector_name}")

    # Replace your existing find_location method with this:
    def find_location(self, satellite_image, drone_image):
        try:
            # Preprocess images
            sat_processed = self.preprocess_image(satellite_image)
            drone_processed = self.preprocess_image(drone_image)

            # Detect keypoints and descriptors on processed images
            kp1, des1 = self.detector.detectAndCompute(sat_processed, None)
            kp2, des2 = self.detector.detectAndCompute(drone_processed, None)

            if kp1 is None or kp2 is None or len(kp1) == 0 or len(kp2) == 0:
                raise ValueError(f"No keypoints found")

            # Match features with adaptive parameters
            if self.detector_name == "sift":
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
                search_params = dict(checks=150)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
                matches = matcher.knnMatch(des1, des2, k=2)
                
                noise_level = self.estimate_noise(drone_image)
                ratio_threshold = min(0.85, 0.75 + noise_level * 0.1)
                
                good_matches = []
                for m, n in matches:
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
                matches = good_matches
            else:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                num_matches = max(4, int(len(matches) * 0.6))
                matches = matches[:num_matches]

            if len(matches) < 4:
                raise ValueError(f"Not enough matches found: {len(matches)}")

            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(
                dst_pts, src_pts, 
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
                maxIters=2500,
                confidence=0.99
            )
            
            if H is None:
                raise ValueError("Could not compute homography")

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
        self.position_error = PositionError()  # Add this line
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
                "transform": None,
                "position": (250, 350)  # Example position - adjust based on your images
            },
            "rotated": {
                "description": "Drone image rotated 45 degrees",
                "transform": lambda img: self.rotate_image(img, 45),
                "position": (250, 350)  # Should be same as normal for rotated image
            },
            "scaled": {
                "description": "Drone image scaled to 80%",
                "transform": lambda img: cv2.resize(img, None, fx=0.8, fy=0.8),
                "position": (250, 350)  # Should be same as normal for scaled image
            },
            "brightness": {
                "description": "Increased brightness",
                "transform": lambda img: cv2.convertScaleAbs(img, alpha=1.5, beta=0),
                "position": (250, 350)  # Should be same as normal for brightness change
            },
            "noise": {
                "description": "Added Gaussian noise",
                "transform": self.add_noise,
                "position": (250, 350)  # Should be same as normal for noisy image
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
                    description=f"{base_name}: {config['description']}",
                    actual_position=config["position"]  # Add actual position
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
            
            # Calculate position error
            error = self.position_error.calculate_error(position, test_case.actual_position)
            self.position_error.add_error(detector_name, test_case.name, position, test_case.actual_position)

            result = {
                'detector': detector_name,
                'test_case': test_case.name,
                'description': test_case.description,
                'computation_time': computation_time,
                'matches_count': matches_count,
                'match_quality': match_quality,
                'position_x': position[0],
                'position_y': position[1],
                'position_error': error,
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
                'position_error': None,
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
        
        # Generate position error visualization and report
        evaluator.position_error.visualize_errors(evaluator.satellite_image)
        error_stats = evaluator.position_error.generate_error_report()
        print("\nPosition Error Statistics:")
        print(error_stats)
        
        # Generate regular report
        evaluator.generate_report()
        
        print("Evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
