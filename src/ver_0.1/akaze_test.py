from main_script import FeatureMatcher, DroneLocalizationEvaluator

evaluator = DroneLocalizationEvaluator(
    satellite_image_path="satellite_image.jpg",
    drone_image_path="drone_image.jpg"
)
evaluator.detectors = ["akaze"]
evaluator.evaluate_detectors()
evaluator.analyze_results()
