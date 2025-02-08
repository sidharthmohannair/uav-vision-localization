from main_script import HybridFeatureMatcher, DroneLocalizationEvaluator

evaluator = DroneLocalizationEvaluator(
    satellite_image_path="satellite_image.jpg",
    drone_image_path="drone_image.jpg"
)
evaluator.detectors = ["hfm"]
evaluator.evaluate_detectors()
evaluator.analyze_results()
