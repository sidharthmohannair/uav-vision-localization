# Legacy Implementations - UAV Vision Localization

## Branch Overview
This branch preserves the historical development path of our UAV vision-based localization system. Each version represents significant architectural decisions, algorithm improvements, and feature additions that have shaped the system's evolution.

## Version History and Key Developments

### Version 1.0.0
Initial implementation:
- Basic feature detectors (SIFT, ORB, AKAZE, BRISK)
- Simple testing framework
- Basic visualization capabilities
- Fundamental matching algorithms

## Directory Structure
```bash
src/
├── visul_drone_locl_TestSuite_v1.py   # Initial implementation
```

## Running Legacy Versions

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements_legacy.txt
```

### Execution
Each version can be run independently:
```bash
python visul_drone_locl_TestSuite_v<version>.py
```

### Test Results
Results for each version are preserved in corresponding directories:
```bash
results/
├── v1_results/
├── v2_results/
└── ...
```

## Implementation Notes
- Each version builds upon lessons learned from previous implementations
- Code comments explain key algorithmic decisions
- Test results demonstrate incremental improvements
- Documentation captures design evolution

## Transition to Modern Implementation
The latest production version has been restructured into a modular architecture. See the `main` branch for the current implementation.

## Historical Documentation
- [Original Design Documents](docs/legacy/design/)
- [Test Reports](docs/legacy/reports/)
- [Performance Comparisons](docs/legacy/benchmarks/)
- [Development Notes](docs/legacy/notes/)