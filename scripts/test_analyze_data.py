import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_pipeline.ml_pipeline_controller import MLPipelineController


def run_test():

    print("\n===== ML PIPELINE TEST STARTED =====\n")

    dataset_path = "datasets/sample_dataset.csv"

    pipeline = MLPipelineController(dataset_path)

    result = pipeline.run_pipeline()

    print("\n===== DATASET ANALYSIS =====\n")
    print(result["analysis"])

    print("\n===== TRAINING RESULTS =====\n")
    print(result["training"])

    print("\n===== EVALUATION REPORT =====\n")
    print(result["evaluation"])

    print("\n===== TEST COMPLETED =====\n")


if __name__ == "__main__":
    run_test()