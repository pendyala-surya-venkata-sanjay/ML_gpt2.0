from ml_pipeline.analyze_data import DatasetAnalyzer
from ml_pipeline.visualization import VisualizationEngine
from ml_pipeline.preprocessing_engine import PreprocessingEngine
from ml_pipeline.feature_engineering import FeatureEngineering
from ml_pipeline.training_engine import TrainingEngine
from ml_pipeline.evaluation_engine import EvaluationEngine
from ml_pipeline.pipeline_genartor import PipelineGenerator
from ml_pipeline.export_project import ProjectExporter

import pandas as pd


class MLPipelineController:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def run_pipeline(self):

        logs = []

        # -------------------------
        # Dataset Analysis
        # -------------------------
        analyzer = DatasetAnalyzer(self.dataset_path)

        analysis_result = analyzer.analyze()

        logs.append("Dataset analysis completed.")
        logs.append(f"Rows: {analysis_result['rows']}")
        logs.append(f"Columns: {analysis_result['columns']}")
        logs.append(f"Target column detected: {analysis_result['target_column']}")
        logs.append(f"Problem type: {analysis_result['problem_type']}")

        target = analysis_result["target_column"]
        problem_type = analysis_result["problem_type"]

        # -------------------------
        # Load Dataset
        # -------------------------
        df = pd.read_csv(self.dataset_path)

        # -------------------------
        # Visualization
        # -------------------------
        viz = VisualizationEngine(df, target)

        visualizations = viz.generate_all()

        logs.append("Visualizations generated.")

        # -------------------------
        # Feature Engineering
        # -------------------------
        fe = FeatureEngineering(df)

        df = fe.apply_all()

        logs.append("Feature engineering applied.")

        # -------------------------
        # Preprocessing
        # -------------------------
        preprocessor = PreprocessingEngine(df, target)

        preprocess_result = preprocessor.preprocess()

        X = preprocess_result["X"]
        y = preprocess_result["y"]

        logs.append("Data preprocessing completed.")

        # -------------------------
        # Model Training
        # -------------------------
        trainer = TrainingEngine(
            X,
            y,
            problem_type,
            feature_engineer=fe,
            preprocessor=preprocessor
        )

        training_result = trainer.train()

        logs.append(f"Best model selected: {training_result['best_model']}")

        best_model = trainer.best_model

        # -------------------------
        # Evaluation
        # -------------------------
        evaluator = EvaluationEngine(
            best_model,
            trainer.X_test,
            trainer.y_test,
            problem_type
        )

        evaluation_result = evaluator.evaluate()

        logs.append("Model evaluation completed.")

        # -------------------------
        # Pipeline Code Generation
        # -------------------------
        generator = PipelineGenerator(
            self.dataset_path,
            target,
            problem_type,
            training_result["best_model"]
        )

        pipeline_file = generator.save_pipeline()

        logs.append("Pipeline code generated successfully.")

        # -------------------------
        # Export Project
        # -------------------------
        exporter = ProjectExporter()

        export_path = exporter.export_project()

        logs.append("Project exported successfully.")
        logs.append(f"Download project at: {export_path}")

        # -------------------------
        # Final Output
        # -------------------------
        return {
            "logs": logs,
            "analysis": analysis_result,
            "visualizations": visualizations,
            "training": training_result,
            "evaluation": evaluation_result,
            "pipeline_code": pipeline_file,
            "export_path": export_path
        }