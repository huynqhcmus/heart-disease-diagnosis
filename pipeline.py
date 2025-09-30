"""
Preprocessing Pipeline for Heart Disease Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

class HeartDiseasePipeline:
    """
    Preprocessing pipeline for heart disease prediction models.
    Handles data loading, preprocessing, and model prediction.
    """
    
    def __init__(self):
        self.preprocessor = None
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.is_fitted = False
        
    def create_preprocessing_pipeline(self):
        """Create the preprocessing pipeline"""
        # Define feature types
        numeric_cols = ['age', 'trestbpd', 'chol', 'thalach', 'oldpeak']
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        # Create preprocessing pipelines
        cat_proc = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', MinMaxScaler())
        ])
        
        num_proc = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocess = ColumnTransformer([
            ('num', num_proc, numeric_cols),
            ('cat', cat_proc, categorical_cols),
        ])
        
        return preprocess
    
    def load_training_data(self, data_path="data/processed/cleveland.csv"):
        """Load training data to fit the preprocessor"""
        try:
            if os.path.exists(data_path):
                # Define column names for cleveland.csv
                column_names = ['age', 'sex', 'cp', 'trestbpd', 'chol', 'fbs', 'restecg', 
                               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
                df = pd.read_csv(data_path, names=column_names)
                feature_cols = ['age', 'sex', 'cp', 'trestbpd', 'chol', 'fbs', 'restecg', 
                               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
                return df[feature_cols]
            return None
        except Exception as e:
            print(f"Error loading training data: {str(e)}")
            return None
    
    def load_models(self, models_dir="models/saved_models"):
        """Load all trained models"""
        try:
            # Model mapping
            model_files = {
                'Random Forest': 'best_random_forest.joblib',
                'Logistic Regression': 'best_logistic_regression.joblib', 
                'K-Nearest Neighbors': 'best_knn.joblib',
                'Decision Tree': 'best_decision_tree.joblib',
                'AdaBoost': 'best_adaboost.joblib',
                'Gradient Boosting': 'best_gradient_boosting.joblib',
                'XGBoost': 'best_xgboost.joblib'
            }
            
            # Load models and their scalers
            for model_name, model_file in model_files.items():
                model_path = os.path.join(models_dir, model_file)
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    
                    # Load scaler if exists
                    scaler_file = model_file.replace('.joblib', '_scaler.joblib')
                    scaler_path = os.path.join(models_dir, scaler_file)
                    if os.path.exists(scaler_path):
                        self.scalers[model_name] = joblib.load(scaler_path)
            
            return len(self.models) > 0
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    def load_metrics(self, models_dir="models/saved_models"):
        """Load model performance metrics"""
        try:
            summary_path = os.path.join(models_dir, 'best_models_summary.json')
            if os.path.exists(summary_path):
                import json
                with open(summary_path, 'r') as f:
                    summary_data = json.load(f)
                    for model_name, info in summary_data['models'].items():
                        self.metrics[model_name] = {
                            'accuracy': info['test_accuracy'],
                            'validation_accuracy': info['validation_accuracy']
                        }
                return True
            return False
        except Exception as e:
            print(f"Error loading metrics: {str(e)}")
            return False
    
    def fit_pipeline(self, data_path="data/processed/cleveland.csv"):
        """Fit the preprocessing pipeline with training data"""
        try:
            # Load training data
            training_data = self.load_training_data(data_path)
            if training_data is None:
                return False
            
            # Create and fit preprocessor
            self.preprocessor = self.create_preprocessing_pipeline()
            self.preprocessor.fit(training_data)
            self.is_fitted = True
            
            return True
        except Exception as e:
            print(f"Error fitting pipeline: {str(e)}")
            return False
    
    def predict(self, input_data):
        """Make predictions using all models"""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit_pipeline() first.")
        
        if len(self.models) == 0:
            raise ValueError("No models loaded. Call load_models() first.")
        
        try:
            # Preprocess input data
            processed_input = self.preprocessor.transform(input_data)
            
            # Make predictions
            results = []
            predictions = []
            
            for model_name, model in self.models.items():
                try:
                    # Make prediction
                    prediction = model.predict(processed_input)[0]
                    predictions.append(prediction)
                    prediction_proba = model.predict_proba(processed_input)[0]
                    confidence = prediction_proba[prediction]
                    
                    results.append({
                        "Model": model_name,
                        "Prediction": "High Risk" if prediction == 1 else "Low Risk",
                        "Confidence": confidence
                    })
                except Exception as e:
                    print(f"Error with {model_name}: {str(e)}")
                    continue
            
            return results, predictions
            
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return [], []
    
    def get_majority_vote(self, predictions):
        """Get majority vote from all predictions"""
        high_risk_votes = sum(p == 1 for p in predictions)
        low_risk_votes = sum(p == 0 for p in predictions)
        
        if high_risk_votes > low_risk_votes:
            return "High Risk", high_risk_votes, len(predictions)
        else:
            return "Low Risk", low_risk_votes, len(predictions)
    
    def initialize(self, models_dir="models/saved_models", data_path="data/processed/cleveland.csv"):
        """Initialize the complete pipeline"""
        print("Initializing Heart Disease Prediction Pipeline...")
        
        # Load models
        if not self.load_models(models_dir):
            print("Failed to load models")
            return False
        
        # Load metrics
        self.load_metrics(models_dir)
        
        # Fit pipeline
        if not self.fit_pipeline(data_path):
            print("Failed to fit pipeline")
            return False
        
        print(f"Pipeline initialized successfully with {len(self.models)} models")
        return True

# Global pipeline instance
pipeline = HeartDiseasePipeline()
