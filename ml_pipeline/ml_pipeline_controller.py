from ml_pipeline.analyze_data import DatasetAnalyzer
from ml_pipeline.visualization import VisualizationEngine
from ml_pipeline.preprocessing_engine import PreprocessingEngine
from ml_pipeline.feature_engineering import FeatureEngineering
from ml_pipeline.training_engine import TrainingEngine
from ml_pipeline.evaluation_engine import EvaluationEngine
from ml_pipeline.pipeline_genartor import PipelineGenerator
from ml_pipeline.export_project import ProjectExporter

import pandas as pd
import json
import os
from typing import Dict, Any, Optional
import numpy as np


class MLPipelineController:
    """Enhanced ML Pipeline with intelligent dataset analysis and user guidance."""

    def __init__(self, dataset_path, user_instructions: Optional[str] = None):
        self.dataset_path = dataset_path
        self.user_instructions = user_instructions
        self.logs = []

    def run_pipeline(self, smart_mode: bool = True):
        """Run the complete ML pipeline with intelligent analysis."""
        
        # -------------------------
        # Enhanced Dataset Analysis
        # -------------------------
        self.logs.append("Starting intelligent dataset analysis...")
        analyzer = DatasetAnalyzer(self.dataset_path)
        
        # Perform comprehensive analysis
        analysis_result = analyzer.analyze()
        
        # Add intelligence layer
        if smart_mode:
            analysis_result = self._enhance_analysis(analysis_result)
        
        self.logs.append(f"Dataset analyzed: {analysis_result['rows']} rows, {analysis_result['columns']} columns")
        self.logs.append(f"Target column: {analysis_result['target_column']}")
        self.logs.append(f"Problem type: {analysis_result['problem_type']}")
        
        # Check for data quality issues
        quality_issues = self._assess_data_quality(analysis_result)
        if quality_issues:
            self.logs.append(f"Data quality alerts: {', '.join(quality_issues)}")
            analysis_result['quality_issues'] = quality_issues
        
        target = analysis_result["target_column"]
        problem_type = analysis_result["problem_type"]
        
        # -------------------------
        # Load Dataset
        # -------------------------
        df = pd.read_csv(self.dataset_path)
        
        # -------------------------
        # Intelligent Visualization
        # -------------------------
        viz = VisualizationEngine(df, target)
        if hasattr(viz, "generate_relevant_visualizations"):
            visualizations = viz.generate_relevant_visualizations(problem_type)
            self.logs.append("Generated relevant visualizations based on data characteristics.")
        else:
            # Defensive fallback for older/newer runtime variants.
            if hasattr(viz, "generate_all"):
                visualizations = viz.generate_all()
                self.logs.append("generate_relevant_visualizations missing; generated all visualizations instead.")
            else:
                visualizations = {}
                self.logs.append("Visualization methods missing; continuing without visualizations.")
        
        # -------------------------
        # Smart Feature Engineering
        # -------------------------
        fe = FeatureEngineering(df)
        
        # Apply feature engineering based on data characteristics
        if smart_mode:
            if hasattr(fe, "apply_intelligent_engineering"):
                fe.apply_intelligent_engineering(analysis_result)
                self.logs.append("Applied intelligent feature engineering.")
            else:
                fe.apply_all()
                self.logs.append("apply_intelligent_engineering missing; falling back to apply_all().")
        else:
            fe.apply_all()
            self.logs.append("Applied default feature engineering (apply_all).")

        # Prefer get_features() if available; otherwise fall back to the raw dataframe.
        if hasattr(fe, "get_features"):
            df = fe.get_features()
        elif hasattr(fe, "df"):
            df = fe.df
        self.logs.append("Prepared engineered feature dataframe.")
        
        # -------------------------
        # Adaptive Preprocessing
        # -------------------------
        preprocessor = PreprocessingEngine(df, target)

        # Use adaptive preprocessing based on data characteristics when available.
        if hasattr(preprocessor, "adaptive_preprocess"):
            preprocess_result = preprocessor.adaptive_preprocess(analysis_result)
        else:
            # Current `PreprocessingEngine` implements `preprocess()`.
            preprocess_result = preprocessor.preprocess()
        
        X = preprocess_result["X"]
        y = preprocess_result["y"]
        
        self.logs.append("Completed adaptive data preprocessing.")
        
        # -------------------------
        # Smart Model Selection & Training
        # -------------------------
        trainer = TrainingEngine(
            X, y, problem_type,
            feature_engineer=fe,
            preprocessor=preprocessor
        )
        
        # Use intelligent model selection if present; otherwise use default training.
        if hasattr(trainer, "intelligent_train"):
            training_result = trainer.intelligent_train(analysis_result)
        else:
            training_result = trainer.train()
        
        self.logs.append(f"Training completed. Best model: {training_result['best_model']}")
        
        best_model = trainer.best_model
        
        # -------------------------
        # Comprehensive Evaluation
        # -------------------------
        evaluator = EvaluationEngine(
            best_model,
            trainer.X_test,
            trainer.y_test,
            problem_type
        )

        # Use comprehensive evaluation when available; otherwise use default evaluate().
        if hasattr(evaluator, "comprehensive_evaluate"):
            evaluation_result = evaluator.comprehensive_evaluate()
        else:
            evaluation_result = evaluator.evaluate()

        # Normalize evaluation shape for downstream exporters/recommendations.
        if isinstance(evaluation_result, dict) and isinstance(evaluation_result.get("metrics"), dict):
            evaluation_result = evaluation_result["metrics"]
        self.logs.append("Model evaluation completed.")
        
        # -------------------------
        # Enhanced Pipeline Code Generation
        # -------------------------
        generator = PipelineGenerator(
            self.dataset_path,
            target,
            problem_type,
            training_result["best_model"]
        )

        if hasattr(generator, "generate_enhanced_pipeline"):
            pipeline_file = generator.generate_enhanced_pipeline(analysis_result, training_result)
        else:
            # Current `PipelineGenerator` provides `save_pipeline()`.
            pipeline_file = generator.save_pipeline()
        self.logs.append("Generated production-ready pipeline code.")
        
        # -------------------------
        # Smart Export
        # -------------------------
        exporter = ProjectExporter()
        export_path = exporter.create_intelligent_export(
            model_path="models/ml_pipeline.pkl",
            analysis_result=analysis_result,
            training_result=training_result,
            evaluation_result=evaluation_result
        )
        
        self.logs.append(f"Project exported successfully: {export_path}")
        
        # -------------------------
        # Final Enhanced Output
        # -------------------------
        return {
            "logs": self.logs,
            "analysis": analysis_result,
            "visualizations": visualizations,
            "training": training_result,
            "evaluation": evaluation_result,
            "pipeline_code": pipeline_file,
            "export_path": export_path,
            "recommendations": self._generate_recommendations(analysis_result, training_result, evaluation_result)
        }
    
    def _enhance_analysis(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance analysis with intelligent insights."""
        
        # Add data complexity assessment
        df = pd.read_csv(self.dataset_path)
        
        # Feature complexity
        numeric_features = len(df.select_dtypes(include=[np.number]).columns)
        categorical_features = len(df.select_dtypes(include=['object']).columns)
        
        analysis_result['feature_complexity'] = {
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'total_features': len(df.columns) - 1,
            'complexity_score': min(10, (numeric_features * 0.5 + categorical_features * 1.5))
        }
        
        # Data quality indicators
        denom = len(df) * len(df.columns)
        missing_percentage = (df.isnull().sum().sum() / denom) * 100 if denom else 0.0
        analysis_result['data_quality'] = {
            'missing_percentage': missing_percentage,
            'quality_score': max(0, 100 - missing_percentage * 2),
            'needs_cleaning': missing_percentage > 5
        }
        
        return analysis_result
    
    def _assess_data_quality(self, analysis_result: Dict[str, Any]) -> list:
        """Assess data quality and return list of issues."""
        issues = []
        
        if analysis_result.get('data_quality', {}).get('missing_percentage', 0) > 10:
            issues.append("High missing values detected")
        
        if analysis_result.get('feature_complexity', {}).get('complexity_score', 0) > 8:
            issues.append("High dimensional dataset")
        
        if analysis_result.get('rows', 0) < 100:
            issues.append("Small dataset size")
        
        return issues
    
    def _generate_recommendations(self, analysis_result: Dict[str, Any], 
                                training_result: Dict[str, Any], 
                                evaluation_result: Dict[str, Any]) -> list:
        """Generate intelligent recommendations based on results."""
        recommendations = []
        
        # Model performance recommendations
        if evaluation_result.get('accuracy', 0) < 0.7:
            recommendations.append("Consider feature engineering or trying different algorithms")
        
        # Data size recommendations
        if analysis_result.get('rows', 0) < 1000:
            recommendations.append("More data may improve model performance")
        
        # Feature recommendations
        complexity = analysis_result.get('feature_complexity', {})
        if complexity.get('categorical_features', 0) > complexity.get('numeric_features', 0):
            recommendations.append("Consider encoding strategies for categorical features")
        
        # Quality recommendations
        if analysis_result.get('data_quality', {}).get('needs_cleaning', False):
            recommendations.append("Data cleaning may improve results")
        
        return recommendations