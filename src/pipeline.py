"""
Preprocessing Pipeline for Heart Disease Prediction
Cập nhật để load models từ folder latest/
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import json
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path if needed
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import feature engineering functions needed for unpickling models
from model_functions import fe_basic, fe_enhanced, fe_poly_only

# Make these available in __main__ namespace for pickle compatibility
import __main__
__main__.fe_basic = fe_basic
__main__.fe_enhanced = fe_enhanced
__main__.fe_poly_only = fe_poly_only

class HeartDiseasePipeline:
    """
    Preprocessing pipeline for heart disease prediction models.
    Handles data loading, preprocessing, and model prediction.
    Load models từ models/saved_models/latest/
    """
    
    def __init__(self):
        self.models = {}
        self.model_info = {}
        self.metrics = {}
        self.is_fitted = False
        self.feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                             'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
    def _apply_feature_engineering(self, X):
        """
        Apply basic feature engineering (không dùng polynomial vì models đã được train với basic FE)
        
        Args:
            X (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data sau khi FE
        """
        X = X.copy()
        
        # Đảm bảo tất cả features có mặt
        for col in self.feature_order:
            if col not in X.columns:
                X[col] = 0
        
        # Sắp xếp theo đúng thứ tự
        X = X[self.feature_order]
        
        return X
    
    def load_models(self, models_dir="models/saved_models/latest"):
        """
        Load all trained models từ folder latest/
        Mỗi model được lưu dạng pipeline với preprocessor, fe_func, fs_indices
        """
        try:
            # Model mapping (theo naming convention trong notebook)
            model_files = {
                'Logistic Regression': 'best_lr_model_pipeline.pkl',
                'Random Forest': 'best_rf_model_pipeline.pkl',
                'K-Nearest Neighbors': 'best_knn_model_pipeline.pkl',
                'Decision Tree': 'best_dt_model_pipeline.pkl',
                # 'AdaBoost': 'best_ada_model_pipeline.pkl',  # TODO: Uncomment when model is available
                'Gradient Boosting': 'best_gb_model_pipeline.pkl',
                'Naive Bayes': 'best_nb_model_pipeline.pkl',
                'SVM': 'best_svm_model_pipeline.pkl',
                'Ensemble': 'best_ensemble_model_pipeline.pkl'
            }
            
            # Load models
            for model_name, model_file in model_files.items():
                model_path = os.path.join(models_dir, model_file)
                if os.path.exists(model_path):
                    try:
                        # Load pipeline (chứa model, preprocessor, fe_func, fs_indices, etc.)
                        pipeline_data = joblib.load(model_path)
                        self.models[model_name] = pipeline_data
                        print(f"✅ Loaded {model_name}")
                    except Exception as e:
                        print(f"⚠️  Skipping {model_name}: {str(e)[:80]}...")
                        continue
                else:
                    print(f"⚠️  {model_name} not found at {model_path}")
            
            if len(self.models) > 0:
                self.is_fitted = True
                print(f"\n✅ Successfully loaded {len(self.models)} models")
                return True
            else:
                print("❌ No models loaded")
                return False
                
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_metrics(self, models_dir="models/saved_models/latest"):
        """Load model performance metrics từ best_models_summary.json"""
        try:
            summary_path = os.path.join(models_dir, 'best_models_summary.json')
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary_data = json.load(f)
                    
                    # Map model abbreviations to full names
                    model_name_map = {
                        'lr': 'Logistic Regression',
                        'rf': 'Random Forest',
                        'knn': 'K-Nearest Neighbors',
                        'dt': 'Decision Tree',
                        # 'ada': 'AdaBoost',  # TODO: Uncomment when model is available
                        'gb': 'Gradient Boosting',
                        'nb': 'Naive Bayes',
                        'svm': 'SVM',
                        'ensemble': 'Ensemble'
                    }
                    
                    # Load metrics for each model
                    if 'best_models' in summary_data:
                        for abbrev, info in summary_data['best_models'].items():
                            full_name = model_name_map.get(abbrev, abbrev)
                            self.metrics[full_name] = {
                                'cv_auc': info.get('cv_auc', 0),
                                'test_auc': info.get('test_auc', 0),
                                'configuration': info.get('configuration', '')
                            }
                    
                    print(f"✅ Loaded metrics for {len(self.metrics)} models")
                    return True
            else:
                print(f"⚠️  Metrics file not found: {summary_path}")
                return False
        except Exception as e:
            print(f"❌ Error loading metrics: {str(e)}")
            return False
    
    def predict(self, input_data):
        """
        Make predictions using all loaded models
        Mỗi model có own preprocessor và feature selection
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call initialize() first.")
        
        if len(self.models) == 0:
            raise ValueError("No models loaded. Call load_models() first.")
        
        try:
            # Apply feature engineering
            X_fe = self._apply_feature_engineering(input_data)
            
            # Make predictions
            results = []
            predictions = []
            
            for model_name, pipeline_data in self.models.items():
                try:
                    # Extract components từ pipeline
                    model = pipeline_data['model']
                    preprocessor = pipeline_data['preprocessor']
                    fe_func = pipeline_data['fe_func']
                    fs_indices = pipeline_data['fs_indices']
                    expected_features = pipeline_data.get('feature_names', None)
                    
                    # Apply feature engineering function (nếu có)
                    if fe_func is not None:
                        # If fe_func is a string, convert to actual function
                        if isinstance(fe_func, str):
                            if fe_func == 'fe_basic':
                                fe_func = fe_basic
                            elif fe_func == 'fe_enhanced':
                                fe_func = fe_enhanced
                            elif fe_func == 'fe_poly_only':
                                fe_func = fe_poly_only
                        
                        # Now call the function
                        X_transformed, _, _ = fe_func(X_fe)
                    else:
                        X_transformed = X_fe
                    
                    # Ensure correct column order if feature names are available
                    if expected_features is not None and isinstance(X_transformed, pd.DataFrame):
                        # Check if feature_names matches all columns (full list) or just a subset
                        if len(expected_features) == len(X_transformed.columns):
                            # Reorder columns to match training order
                            X_transformed = X_transformed[expected_features]
                        else:
                            # Feature names mismatch (saved incorrectly during retrain)
                            # Convert to numpy to avoid sklearn validation issues
                            X_transformed = X_transformed.values
                    
                    # Apply preprocessing (scaling, etc.)
                    X_preprocessed = preprocessor.transform(X_transformed)
                    
                    # Convert to array if sparse
                    if hasattr(X_preprocessed, 'toarray'):
                        X_preprocessed = X_preprocessed.toarray()
                    
                    # Apply feature selection (if indices provided)
                    if fs_indices is not None:
                        X_selected = X_preprocessed[:, fs_indices]
                    else:
                        # No feature selection, use all features
                        X_selected = X_preprocessed
                    
                    # Make prediction
                    prediction = model.predict(X_selected)[0]
                    predictions.append(prediction)
                    
                    # Get probability
                    if hasattr(model, 'predict_proba'):
                        prediction_proba = model.predict_proba(X_selected)[0]
                        confidence = prediction_proba[prediction]
                    else:
                        # For models without predict_proba (e.g., some SVMs)
                        confidence = 1.0 if prediction == 1 else 0.0
                    
                    results.append({
                        "Model": model_name,
                        "Prediction": "High Risk" if prediction == 1 else "Low Risk",
                        "Confidence": float(confidence)
                    })
                    
                except Exception as e:
                    print(f"❌ Error with {model_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            return results, predictions
            
        except Exception as e:
            print(f"❌ Error making predictions: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def get_majority_vote(self, predictions):
        """Get majority vote from all predictions"""
        high_risk_votes = sum(p == 1 for p in predictions)
        low_risk_votes = sum(p == 0 for p in predictions)
        
        if high_risk_votes > low_risk_votes:
            return "High Risk", high_risk_votes, len(predictions)
        else:
            return "Low Risk", low_risk_votes, len(predictions)
    
    def initialize(self, models_dir="models/saved_models/latest"):
        """
        Initialize the complete pipeline
        Load models từ folder latest/
        """
        print("=" * 80)
        print("🫀 Initializing Heart Disease Prediction Pipeline...")
        print("=" * 80)
        
        # Load models (models đã có preprocessor riêng, không cần fit thêm)
        if not self.load_models(models_dir):
            print("❌ Failed to load models")
            return False
        
        # Load metrics
        self.load_metrics(models_dir)
        
        # Mark as fitted
        self.is_fitted = True
        
        print("=" * 80)
        print(f"✅ Pipeline initialized successfully with {len(self.models)} models")
        print("=" * 80)
        return True

# Global pipeline instance
pipeline = HeartDiseasePipeline()
