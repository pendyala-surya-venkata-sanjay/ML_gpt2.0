import os
import zipfile
import shutil
import json
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd


class ProjectExporter:
    """Enhanced project exporter with comprehensive, production-ready outputs."""

    def __init__(self):
        self.export_folder = "generated_projects"
        self.model_path = "models/ml_pipeline.pkl"
        
        # Ensure export directory exists
        if not os.path.exists(self.export_folder):
            os.makedirs(self.export_folder)

    def create_intelligent_export(self, 
                                model_path: Optional[str] = None, 
                                project_name: Optional[str] = None,
                                analysis_result: Optional[Dict[str, Any]] = None,
                                training_result: Optional[Dict[str, Any]] = None,
                                evaluation_result: Optional[Dict[str, Any]] = None) -> str:
        """Create a comprehensive, intelligent project export."""
        
        # Generate project name
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        project_name = project_name or f"ml_project_{ts}"
        project_path = os.path.join(self.export_folder, project_name)
        
        # Clean up existing project if it exists
        if os.path.exists(project_path):
            shutil.rmtree(project_path)
        os.makedirs(project_path)
        
        # Copy model
        src_model = model_path or self.model_path
        if os.path.exists(src_model):
            shutil.copy(src_model, os.path.join(project_path, "model.pkl"))
        
        # Create comprehensive README
        readme_content = self._generate_readme(
            project_name, analysis_result, training_result, evaluation_result
        )
        with open(os.path.join(project_path, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        # Create requirements.txt
        requirements_content = self._generate_requirements()
        with open(os.path.join(project_path, "requirements.txt"), "w") as f:
            f.write(requirements_content)
        
        # Create enhanced prediction script
        prediction_script = self._generate_prediction_script(analysis_result, training_result)
        with open(os.path.join(project_path, "predict.py"), "w", encoding="utf-8") as f:
            f.write(prediction_script)
        
        # Create training script (reproducible training)
        training_script = self._generate_training_script(analysis_result)
        with open(os.path.join(project_path, "train.py"), "w", encoding="utf-8") as f:
            f.write(training_script)
        
        # Create data analysis script
        analysis_script = self._generate_analysis_script(analysis_result)
        with open(os.path.join(project_path, "analyze.py"), "w", encoding="utf-8") as f:
            f.write(analysis_script)
        
        # Create model metadata
        metadata_content = self._generate_metadata(
            analysis_result, training_result, evaluation_result
        )
        with open(os.path.join(project_path, "model_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self._sanitize_for_json(metadata_content), f, indent=2, ensure_ascii=False)
        
        # Create configuration file
        config_content = self._generate_config(analysis_result)
        with open(os.path.join(project_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_content, f, indent=2, ensure_ascii=False)
        
        # Create example data if available
        if analysis_result and 'sample_data' in analysis_result:
            self._create_sample_data(project_path, analysis_result)
        
        # Create utility scripts
        self._create_utility_scripts(project_path)
        
        # Create the final zip file
        zip_path = os.path.join(self.export_folder, f"{project_name}.zip")
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(project_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, project_path)
                    zipf.write(file_path, arcname)
        
        return zip_path

    def _sanitize_for_json(self, obj: Any) -> Any:
        """
        Ensure the export payload is JSON serializable.

        In particular, some templates may end up carrying `type` objects (e.g. `bool`)
        into metadata. The default `json` encoder cannot serialize those.
        """
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        # Handle type objects like `bool`, `int`, etc.
        if isinstance(obj, type):
            return obj.__name__

        # Optional numpy handling (avoid hard dependency).
        try:
            import numpy as np  # type: ignore

            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except Exception:
            pass

        if isinstance(obj, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple, set)):
            return [self._sanitize_for_json(v) for v in obj]

        # Fallback: stringify unknown objects.
        return str(obj)
    
    def _generate_readme(self, project_name: str, analysis_result: Dict[str, Any], 
                        training_result: Dict[str, Any], evaluation_result: Dict[str, Any]) -> str:
        """Generate comprehensive README.md file."""
        
        target_column = analysis_result.get('target_column', 'target') if analysis_result else 'target'
        problem_type = analysis_result.get('problem_type', 'classification') if analysis_result else 'classification'
        best_model = training_result.get('best_model', 'Unknown') if training_result else 'Unknown'
        accuracy = evaluation_result.get('accuracy', 0) if evaluation_result else 0
        
        readme = f"""# {project_name}

## Overview
This is a production-ready machine learning project generated by ML Assistant. 
The project includes a trained model, preprocessing pipeline, and all necessary components for deployment.

## Model Information
- **Problem Type**: {problem_type.title()}
- **Target Column**: {target_column}
- **Best Model**: {best_model}
- **Accuracy**: {accuracy:.4f}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Project Structure
```
{project_name}/
├── model.pkl              # Trained machine learning model
├── predict.py              # Prediction script
├── train.py                # Training reproduction script
├── analyze.py              # Data analysis script
├── requirements.txt        # Python dependencies
├── config.json            # Model configuration
├── model_metadata.json   # Detailed model information
├── utils.py               # Utility functions
└── README.md              # This file
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Make Predictions
```bash
python predict.py
```

### 3. Retrain the Model
```bash
python train.py
```

### 4. Analyze Your Data
```bash
python analyze.py
```

## Usage

### Prediction
The `predict.py` script provides an easy way to make predictions:

```python
from predict import MLPredictor

# Load the model
predictor = MLPredictor()

# Make a prediction
sample_data = {{
    'feature1': value1,
    'feature2': value2,
    # ... more features
}}

prediction = predictor.predict(sample_data)
print(f"Prediction: {{prediction}}")
```

### Training
To retrain the model with new data:

```python
from train import MLTrainer

# Initialize trainer
trainer = MLTrainer('your_data.csv')

# Train the model
results = trainer.train()
print(f"Model trained with accuracy: {{results['accuracy']}}")
```

## Model Details

### Data Characteristics
{self._format_data_info(analysis_result) if analysis_result else 'No analysis data available'}

### Performance Metrics
{self._format_performance_metrics(evaluation_result) if evaluation_result else 'No evaluation data available'}

## Requirements

See `requirements.txt` for the complete list of dependencies. Key packages include:
- pandas
- scikit-learn
- joblib
- numpy

## Configuration

Model configuration is stored in `config.json`. You can modify settings such as:
- Feature preprocessing parameters
- Model hyperparameters
- Training options

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Not Found**: Ensure `model.pkl` exists in the project directory

3. **Data Format Issues**: Check that your input data matches the expected format

### Support

For issues and questions, refer to the generated scripts or contact your ML team.

## License

This project was generated by ML Assistant. Please review the terms of use for your specific deployment.
"""
        return readme
    
    def _format_data_info(self, analysis_result: Dict[str, Any]) -> str:
        """Format data information for README."""
        info = []
        
        if 'rows' in analysis_result:
            info.append(f"- Dataset Size: {analysis_result['rows']:,} rows")
        
        if 'columns' in analysis_result:
            info.append(f"- Features: {analysis_result['columns'] - 1} columns")
        
        if 'feature_complexity' in analysis_result:
            complexity = analysis_result['feature_complexity']
            info.append(f"- Numeric Features: {complexity.get('numeric_features', 0)}")
            info.append(f"- Categorical Features: {complexity.get('categorical_features', 0)}")
        
        if 'data_quality' in analysis_result:
            quality = analysis_result['data_quality']
            info.append(f"- Data Quality Score: {quality.get('quality_score', 0):.1f}/100")
        
        return '\n'.join(info) if info else '- No detailed data information available'
    
    def _format_performance_metrics(self, evaluation_result: Dict[str, Any]) -> str:
        """Format performance metrics for README."""
        metrics = []
        
        for key, value in evaluation_result.items():
            if isinstance(value, (int, float)):
                if key == 'accuracy':
                    metrics.append(f"- Accuracy: {value:.4f}")
                elif 'f1' in key.lower():
                    metrics.append(f"- {key.title()}: {value:.4f}")
                elif 'precision' in key.lower():
                    metrics.append(f"- Precision: {value:.4f}")
                elif 'recall' in key.lower():
                    metrics.append(f"- Recall: {value:.4f}")
                else:
                    metrics.append(f"- {key.title()}: {value:.4f}")
        
        return '\n'.join(metrics) if metrics else '- No performance metrics available'
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt with specific versions."""
        return """pandas>=1.5.0
scikit-learn>=1.2.0
numpy>=1.21.0
joblib>=1.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
"""
    
    def _generate_prediction_script(self, analysis_result: Dict[str, Any], 
                                   training_result: Dict[str, Any]) -> str:
        """Generate enhanced prediction script."""
        
        target_column = analysis_result.get('target_column', 'target') if analysis_result else 'target'
        problem_type = analysis_result.get('problem_type', 'classification') if analysis_result else 'classification'
        
        script = '''#!/usr/bin/env python3
"""
Prediction script for __TARGET_COLUMN__ __PROBLEM_TYPE__ model.
Generated by ML Assistant.
"""

import joblib
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Union
import sys
import os


class MLPredictor:
    """Machine Learning Predictor for __PROBLEM_TYPE__."""
    
    def __init__(self, model_path: str = "model.pkl", config_path: str = "config.json"):
        """Initialize the predictor."""
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.feature_columns = None
        
        self._load_model()
        self._load_config()
    
    def _load_model(self):
        """Load the trained model."""
        try:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    def _load_config(self):
        """Load model configuration."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            self.feature_columns = self.config.get('feature_columns', [])
        except FileNotFoundError:
            print("Warning: Config file not found, using default settings")
            self.config = {{}}
            self.feature_columns = []
    
    def _preprocess_input(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """Preprocess input data."""
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Ensure all required features are present
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0  # Default value for missing features
            
            # Reorder columns to match training
            df = df[self.feature_columns]
        
        return df
    
    def predict(self, data: Union[Dict, pd.DataFrame]) -> Any:
        """
        Make predictions on new data.
        
        Args:
            data: Input data as dict or DataFrame
            
        Returns:
            Prediction result
        """
        # Preprocess input
        processed_data = self._preprocess_input(data)
        
        # Make prediction
        try:
            prediction = self.model.predict(processed_data)
            
            # For classification, also return probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)
                return {
                    'prediction': prediction[0] if len(prediction) == 1 else prediction,
                    'probabilities': probabilities[0] if len(probabilities) == 1 else probabilities
                }
            else:
                return {
                    'prediction': prediction[0] if len(prediction) == 1 else prediction
                }
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def predict_batch(self, data: Union[Dict, pd.DataFrame]) -> Any:
        """Make predictions on multiple samples."""
        return self.predict(data)


def main():
    """Main function for command-line prediction."""
    
    # Example usage
    predictor = MLPredictor()
    
    # Example data - replace with your actual data
    example_data = {{
        # Add your feature values here
        # Example:
        # 'feature1': 10,
        # 'feature2': 'category',
        # ...
    }}
    
    if not example_data or len(example_data) <= 1:
        print("Error: Please update the example_data in the main() function with your actual feature values.")
        print("Refer to the config.json file for expected feature names.")
        sys.exit(1)
    
    try:
        result = predictor.predict(example_data)
        print("Prediction Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error making prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
        script = script.replace('__TARGET_COLUMN__', str(target_column)).replace('__PROBLEM_TYPE__', str(problem_type))
        return script
    
    def _generate_training_script(self, analysis_result: Dict[str, Any]) -> str:
        """Generate training reproduction script."""
        target_column = analysis_result.get('target_column', 'target') if analysis_result else 'target'
        
        script = '''#!/usr/bin/env python3
"""
Training script for reproducing the model.
Generated by ML Assistant.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os


class MLTrainer:
    """Machine Learning Trainer for __TARGET_COLUMN__ prediction."""
    
    def __init__(self, data_path: str, target_column: str = "__TARGET_COLUMN__"):
        """Initialize the trainer."""
        self.data_path = data_path
        self.target_column = target_column
        self.df = None
        self.X = None
        self.y = None
        self.model = None
        self.scaler = None
        self.label_encoders = {{}}
        self.feature_columns = []
    
    def load_data(self):
        """Load and prepare the data."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded: {self.df.shape}")
            
            if self.target_column not in self.df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in data")
                
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")
    
    def preprocess_data(self):
        """Preprocess the data for training."""
        # Separate features and target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Handle categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].fillna('Unknown'))
            else:
                X[col] = self.label_encoders[col].transform(X[col].fillna('Unknown'))
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Encode target if it's categorical
        if y.dtype == 'object':
            if 'target' not in self.label_encoders:
                self.label_encoders['target'] = LabelEncoder()
                y = self.label_encoders['target'].fit_transform(y)
            else:
                y = self.label_encoders['target'].transform(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
    
    def train_model(self):
        """Train the model."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed")
    
    def evaluate_model(self):
        """Evaluate the model."""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
        }
    
    def save_model(self, model_path: str = "model.pkl"):
        """Save the trained model and preprocessing objects."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def save_config(self, config_path: str = "config.json"):
        """Save model configuration."""
        config = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_type': 'RandomForestClassifier',
            'preprocessing': {
                'scaling': True,
                'label_encoding': list(self.label_encoders.keys())
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to {config_path}")
    
    def train(self, save_model: bool = True):
        """Run the complete training pipeline."""
        self.load_data()
        self.preprocess_data()
        self.train_model()
        results = self.evaluate_model()
        
        if save_model:
            self.save_model()
            self.save_config()
        
        return results


def main():
    """Main function for training."""
    if len(sys.argv) != 2:
        print("Usage: python train.py <data_file.csv>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found")
        sys.exit(1)
    
    try:
        trainer = MLTrainer(data_file)
        results = trainer.train()
        print(f"\nTraining completed successfully!")
        print(f"Final accuracy: {results['accuracy']:.4f}")
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
        script = script.replace('__TARGET_COLUMN__', str(target_column))
        return script
    
    def _generate_analysis_script(self, analysis_result: Dict[str, Any]) -> str:
        """Generate data analysis script."""
        return '''#!/usr/bin/env python3
"""
Data analysis script for exploring datasets.
Generated by ML Assistant.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import sys
import os


class DataAnalyzer:
    """Data analysis and visualization toolkit."""
    
    def __init__(self, data_path: str):
        """Initialize the analyzer."""
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """Load the dataset."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded: {self.df.shape}")
            print(f"\\nColumns: {list(self.df.columns)}")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")
    
    def basic_info(self):
        """Display basic dataset information."""
        print("\\n=== Dataset Info ===")
        print(self.df.info())
        
        print("\\n=== Descriptive Statistics ===")
        print(self.df.describe())
        
        print("\\n=== Missing Values ===")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.any() else "No missing values")
    
    def visualize_distributions(self, save_plots: bool = True):
        """Create distribution plots for numeric columns."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return
        
        # Create subplots
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        
        for i, col in enumerate(numeric_cols):
            row, col_idx = i // n_cols, i % n_cols
            ax = axes[row][col_idx] if n_rows > 1 else axes[col_idx]
            
            self.df[col].hist(bins=30, ax=ax)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(numeric_cols), n_rows * n_cols):
            row, col_idx = i // n_cols, i % n_cols
            if n_rows > 1:
                axes[row][col_idx].set_visible(False)
            elif n_cols > 1:
                axes[col_idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
            print("Distributions plot saved as 'distributions.png'")
        
        plt.show()
    
    def correlation_analysis(self, save_plots: bool = True):
        """Create correlation heatmap."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("Need at least 2 numeric columns for correlation analysis")
            return
        
        correlation_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print("Correlation heatmap saved as 'correlation_heatmap.png'")
        
        plt.show()
    
    def target_analysis(self, target_column: str, save_plots: bool = True):
        """Analyze target variable distribution."""
        if target_column not in self.df.columns:
            print(f"Target column '{target_column}' not found")
            return
        
        target = self.df[target_column]
        
        plt.figure(figsize=(10, 6))
        
        if target.dtype == 'object':
            # Categorical target
            value_counts = target.value_counts()
            plt.bar(value_counts.index, value_counts.values)
            plt.xlabel(target_column)
            plt.ylabel('Count')
            plt.title(f'Distribution of {target_column}')
            plt.xticks(rotation=45)
        else:
            # Numeric target
            plt.hist(target, bins=30, alpha=0.7)
            plt.xlabel(target_column)
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {target_column}')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
            print("Target distribution saved as 'target_distribution.png'")
        
        plt.show()
    
    def full_analysis(self, target_column: str = None):
        """Run complete analysis pipeline."""
        self.load_data()
        self.basic_info()
        
        print("\\n=== Generating Visualizations ===")
        self.visualize_distributions()
        self.correlation_analysis()
        
        if target_column:
            self.target_analysis(target_column)
        
        print("\\nAnalysis completed!")


def main():
    """Main function for analysis."""
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <data_file.csv> [target_column]")
        sys.exit(1)
    
    data_file = sys.argv[1]
    target_column = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found")
        sys.exit(1)
    
    try:
        analyzer = DataAnalyzer(data_file)
        analyzer.full_analysis(target_column)
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
    
    def _generate_metadata(self, analysis_result: Dict[str, Any], 
                          training_result: Dict[str, Any], 
                          evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model metadata."""
        metadata = {
            "project_info": {
                "generated_by": "ML Assistant",
                "generated_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "data_info": analysis_result or {},
            "training_info": training_result or {},
            "evaluation_info": evaluation_result or {},
            "model_info": {
                "model_type": training_result.get('best_model', 'Unknown') if training_result else 'Unknown',
                "problem_type": analysis_result.get('problem_type', 'Unknown') if analysis_result else 'Unknown',
                "target_column": analysis_result.get('target_column', 'Unknown') if analysis_result else 'Unknown'
            }
        }
        return metadata
    
    def _generate_config(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration file."""
        config = {
            "model_settings": {
                "problem_type": analysis_result.get('problem_type', 'classification') if analysis_result else 'classification',
                "target_column": analysis_result.get('target_column', 'target') if analysis_result else 'target'
            },
            "preprocessing": {
                "handle_missing": True,
                "scale_features": True,
                "encode_categorical": True
            },
            "feature_columns": analysis_result.get('feature_columns', []) if analysis_result else [],
            "model_parameters": {
                "random_state": 42,
                "n_estimators": 100
            }
        }
        return config
    
    def _create_sample_data(self, project_path: str, analysis_result: Dict[str, Any]):
        """Create sample data file for testing."""
        # This would create a small sample of the original data
        # Implementation depends on having access to sample data
        pass
    
    def _create_utility_scripts(self, project_path: str):
        """Create utility scripts."""
        utils_script = '''#!/usr/bin/env python3
"""
Utility functions for the ML project.
Generated by ML Assistant.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def validate_input_data(data: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Validate input data against configuration."""
    required_features = config.get('feature_columns', [])
    
    if not required_features:
        return True
    
    missing_features = [feat for feat in required_features if feat not in data]
    
    if missing_features:
        print(f"Missing required features: {missing_features}")
        return False
    
    return True


def prepare_input_data(data: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    """Prepare input data for prediction."""
    feature_columns = config.get('feature_columns', [])
    
    # Create DataFrame with required columns
    df = pd.DataFrame([data])
    
    # Add missing columns with default values
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns
    if feature_columns:
        df = df[feature_columns]
    
    return df


def format_prediction_output(prediction: Any, probabilities: Any = None) -> Dict[str, Any]:
    """Format prediction output for consistent API."""
    result = {
        "prediction": prediction,
        "timestamp": pd.Timestamp.now().isoformat(),
        "status": "success"
    }
    
    if probabilities is not None:
        result["probabilities"] = probabilities
    
    return result


def log_prediction(input_data: Dict[str, Any], prediction_result: Dict[str, Any], log_file: str = "predictions.log"):
    """Log prediction for audit trail."""
    log_entry = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "input": input_data,
        "output": prediction_result
    }
    
    try:
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Warning: Could not log prediction: {e}")


if __name__ == "__main__":
    print("Utility functions loaded. Import this module to use the functions.")
'''
        
        with open(os.path.join(project_path, "utils.py"), "w", encoding="utf-8") as f:
            f.write(utils_script)

    # Legacy method for compatibility
    def export_project(self, model_path=None, project_name=None):
        """Legacy export method for backward compatibility."""
        return self.create_intelligent_export(
            model_path=model_path, 
            project_name=project_name
        )