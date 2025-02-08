# UAV Vision-Based Localization System

## Project Overview
This project develops a sophisticated vision-based localization system for unmanned aerial vehicles (UAVs) operating in GPS-denied environments. The system enables UAVs to determine their position by matching real-time camera feed with satellite imagery through advanced computer vision techniques.

The system has evolved through multiple iterations, each enhancing detection accuracy, computational efficiency, and robustness across various environmental conditions. The current implementation (v6.2) features a hybrid detection approach that adaptively combines multiple feature detection algorithms to achieve optimal performance.

## Core Capabilities

### Feature Detection and Matching
The system implements multiple feature detection algorithms, each optimized for specific scenarios:
- SIFT (Scale-Invariant Feature Transform) for robust scale and rotation handling
- ORB (Oriented FAST and Rotated BRIEF) for efficient real-time processing
- AKAZE (Accelerated-KAZE) for handling nonlinear scale space
- BRISK (Binary Robust Invariant Scalable Keypoints) for fast binary descriptors
- Hybrid detector (v6.0+) combining multiple algorithms for optimal performance

### Advanced Processing Pipeline
Our pipeline incorporates several sophisticated techniques:
- Adaptive preprocessing for varying lighting conditions
- Region of Interest (ROI) optimization for efficient processing
- Multi-stage matching with outlier rejection
- Comprehensive error analysis and visualization
- Scale and rotation invariant position estimation

### Testing Framework
The system includes a robust testing framework that evaluates performance across:
- Multiple environmental conditions
- Various image transformations (rotation, scale, brightness)
- Different noise levels and distortions
- Real-world deployment scenarios

## Project Structure
```bash
uav-vision-localization/
├── src/                      # Source code
│   ├── core/                 # Core detection algorithms
│   ├── utils/                # Utility functions
│   └── evaluation/           # Testing framework
├── tests/                    # Test suites
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
├── datasets/                 # Test datasets
│   ├── satellite/            # Satellite imagery
│   └── drone/               # Drone camera feeds
├── results/                  # Evaluation results
├── docs/                     # Documentation
└── scripts/                  # Utility scripts
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- OpenCV 4.5+
- NumPy
- Matplotlib
- Pandas

### Installation
```bash
# Clone the repository
git clone https://github.com/sidharthmohannair/uav-vision-localization.git
cd uav-vision-localization

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```python
from src.core.evaluator import PracticalDroneEvaluator

# Initialize evaluator
evaluator = PracticalDroneEvaluator(
    satellite_path="datasets/satellite/sample.jpg",
    drone_images=["datasets/drone/sample.jpg"],
    drone_position=(lat, lon)  # Optional ground truth
)

# Run evaluation
results = evaluator.run_evaluation()

# Generate comprehensive report
evaluator.generate_report()
```

## Development Status
![Version](https://img.shields.io/badge/version-6.2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/opencv-4.5%2B-green)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Performance Metrics
Our latest version achieves:
- Position accuracy: <1% of flight height
- Processing time: <100ms per frame
- Match quality: >80% inlier ratio
- Success rate: >95% across test scenarios

## Documentation
- [Algorithm Deep Dive](docs/algorithms/)
- [API Reference](docs/api/)
- [Development History](docs/history.md)
- [Performance Analysis](docs/benchmarks.md)
- [Implementation Guide](docs/implementation.md)

## Contributing
We welcome contributions! See our [Contribution Guidelines](CONTRIBUTING.md) for details on:
- Code style and standards
- Testing requirements
- Pull request process
- Development workflow

## License
This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments
This project builds upon research in computer vision and UAV navigation, particularly:
- Feature detection algorithms (SIFT, ORB, AKAZE, BRISK)
- OpenCV library and community
- Related research in vision-based navigation