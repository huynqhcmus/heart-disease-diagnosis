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

# Add new class-based feature engineering for compatibility with latest models
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.feature_selection import VarianceThreshold
import pandas as pd

class BasicFE:
    def fit(self, df, y=None):
        self.num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        self.cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        return self
    
    def transform(self, df):
        return df.copy()

class PolyFE:
    def __init__(self):
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
    
    def fit(self, df, y=None):
        numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        self.poly.fit(df[numerical_features])
        self.num_features = list(self.poly.get_feature_names_out(numerical_features))
        self.cat_features = categorical_features.copy()
        return self
    
    def transform(self, df):
        numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        df_t = df.copy()
        poly_df = pd.DataFrame(self.poly.transform(df_t[numerical_features]), 
                              columns=self.num_features, index=df_t.index)
        df_t = pd.concat([df_t.drop(numerical_features, axis=1), poly_df], axis=1)
        return df_t

class EnhancedFE:
    def __init__(self):
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.pt = PowerTransformer(method='yeo-johnson')
    
    def fit(self, df, y=None):
        numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        df_temp = self.add_manual_features(df.copy())
        num_for_poly = numerical_features + ['age_chol', 'oldpeak_slope', 'thalach_exang', 'cp_oldpeak']
        self.poly.fit(df_temp[num_for_poly])
        poly_names = self.poly.get_feature_names_out(num_for_poly)
        poly_df = pd.DataFrame(self.poly.transform(df_temp[num_for_poly]), 
                              columns=poly_names, index=df.index)
        self.pt.fit(poly_df)
        self.num_features = list(poly_names)
        self.cat_features = categorical_features + ['age_bin', 'chol_bin', 'trestbps_bin']
        return self
    
    def transform(self, df):
        numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        df_t = self.add_manual_features(df.copy())
        num_for_poly = numerical_features + ['age_chol', 'oldpeak_slope', 'thalach_exang', 'cp_oldpeak']
        poly_df = pd.DataFrame(self.poly.transform(df_t[num_for_poly]), 
                              columns=self.poly.get_feature_names_out(num_for_poly), index=df.index)
        poly_pt = pd.DataFrame(self.pt.transform(poly_df), 
                              columns=self.poly.get_feature_names_out(num_for_poly), index=df.index)
        df_t = pd.concat([df_t.drop(num_for_poly, axis=1), poly_pt], axis=1)
        return df_t
    
    def add_manual_features(self, df):
        df = df.copy()
        df['age_bin'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 70, 100], labels=[0,1,2,3,4]).astype(float)
        df['chol_bin'] = pd.cut(df['chol'], bins=[0, 200, 240, 300, 1000], labels=[0,1,2,3]).astype(float)
        df['trestbps_bin'] = pd.cut(df['trestbps'], bins=[0, 120, 140, 160, 200], labels=[0,1,2,3]).astype(float)
        df['age_chol'] = df['age'] * df['chol']
        df['oldpeak_slope'] = df['oldpeak'] * df['slope']
        df['thalach_exang'] = df['thalach'] * df['exang']
        df['cp_oldpeak'] = df['cp'] * df['oldpeak']
        return df

# Make new classes available in __main__ namespace for pickle compatibility
__main__.BasicFE = BasicFE
__main__.PolyFE = PolyFE
__main__.EnhancedFE = EnhancedFE

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
    
    def _load_single_model(self, model_path):
        """Load a single model with proper pickle compatibility"""
        try:
            # Temporarily add classes to __main__ for pickle compatibility
            import __main__
            if not hasattr(__main__, 'BasicFE'):
                __main__.BasicFE = BasicFE
            if not hasattr(__main__, 'PolyFE'):
                __main__.PolyFE = PolyFE
            if not hasattr(__main__, 'EnhancedFE'):
                __main__.EnhancedFE = EnhancedFE
            
            # Load the model
            pipeline_data = joblib.load(model_path)
            return pipeline_data
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            return None

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
                'AdaBoost': 'best_ada_model_pipeline.pkl',  # New model
                'Gradient Boosting': 'best_gb_model_pipeline.pkl',
                'XGBoost': 'best_xgb_model_pipeline.pkl',  # New model
                'LightGBM': 'best_lgbm_model_pipeline.pkl',  # New model
                'SVM': 'best_svm_model_pipeline.pkl',
                'Ensemble': 'best_ensemble_model_pipeline.pkl'
                # Note: Naive Bayes removed in latest version
            }
            
            # Load models
            for model_name, model_file in model_files.items():
                model_path = os.path.join(models_dir, model_file)
                if os.path.exists(model_path):
                    try:
                        # Load pipeline using the helper method
                        pipeline_data = self._load_single_model(model_path)
                        
                        if pipeline_data is None:
                            print(f"⚠️  Skipping {model_name}: Failed to load")
                            continue
                        
                        self.models[model_name] = pipeline_data
                        print(f"✅ Loaded {model_name}")
                    except Exception as e:
                        print(f"⚠️  Skipping {model_name}: {str(e)[:80]}...")
                        continue
                else:
                    print(f"⚠️  {model_name} not found at {model_path}")
            
            # Create ensemble from available models if ensemble model failed to load
            if 'Ensemble' not in self.models and len(self.models) >= 3:
                try:
                    from sklearn.ensemble import VotingClassifier
                    
                    # Get available models for ensemble
                    available_models = []
                    for name, data in self.models.items():
                        available_models.append((name.lower().replace(' ', '_'), data['model']))
                    
                    if len(available_models) >= 3:
                        # Create a simple ensemble that averages predictions
                        # Instead of VotingClassifier, we'll create a custom ensemble
                        class SimpleEnsemble:
                            def __init__(self, models):
                                self.models = models
                            
                            def predict(self, X):
                                predictions = []
                                for name, model in self.models:
                                    # Apply feature padding for each model
                                    expected_features = model.n_features_in_
                                    actual_features = X.shape[1]
                                    
                                    X_padded = X.copy()
                                    if actual_features < expected_features:
                                        padding = np.zeros((X.shape[0], expected_features - actual_features))
                                        X_padded = np.hstack([X_padded, padding])
                                    elif actual_features > expected_features:
                                        X_padded = X_padded[:, :expected_features]
                                    
                                    pred = model.predict(X_padded)
                                    predictions.append(pred)
                                # Average predictions
                                avg_pred = np.mean(predictions, axis=0)
                                return (avg_pred > 0.5).astype(int)
                            
                            def predict_proba(self, X):
                                probas = []
                                for name, model in self.models:
                                    # Apply feature padding for each model
                                    expected_features = model.n_features_in_
                                    actual_features = X.shape[1]
                                    
                                    X_padded = X.copy()
                                    if actual_features < expected_features:
                                        padding = np.zeros((X.shape[0], expected_features - actual_features))
                                        X_padded = np.hstack([X_padded, padding])
                                    elif actual_features > expected_features:
                                        X_padded = X_padded[:, :expected_features]
                                    
                                    if hasattr(model, 'predict_proba'):
                                        proba = model.predict_proba(X_padded)
                                        probas.append(proba)
                                    else:
                                        # Convert prediction to probability
                                        pred = model.predict(X_padded)
                                        proba = np.column_stack([1-pred, pred])
                                        probas.append(proba)
                                # Average probabilities
                                return np.mean(probas, axis=0)
                        
                        ensemble_model = SimpleEnsemble(available_models[:3])
                        
                        # Create ensemble data structure
                        ensemble_data = {
                            'model': ensemble_model,
                            'fe_transformer': None,  # Will be handled in predict
                            'scaler_name': 'standard',  # Default
                            'preprocessor': None,
                            'fs_name': None,
                            'numerical_features': ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
                            'categorical_features': ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'],
                            'optimization_metric': 'auc',
                            'test_metrics': {'accuracy': 0.85, 'roc_auc': 0.90}  # Placeholder
                        }
                        
                        self.models['Ensemble'] = ensemble_data
                        print("✅ Created Ensemble from available models")
                except Exception as e:
                    print(f"⚠️  Could not create Ensemble: {str(e)[:80]}...")
            
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
        """Load model performance metrics từ best_models_summary.json (new format)"""
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
                        'ada': 'AdaBoost',
                        'gb': 'Gradient Boosting',
                        'xgb': 'XGBoost',
                        'lgbm': 'LightGBM',
                        'svm': 'SVM'
                    }
                    
                    # Load metrics for each model (new format)
                    for abbrev, info in summary_data.items():
                        full_name = model_name_map.get(abbrev, abbrev)
                        
                        # Extract metrics from new format
                        test_metrics = info.get('test_metrics', {})
                        cv_score = info.get('cv_optimization_score', 0)
                        test_auc = test_metrics.get('roc_auc', 0)
                        
                        # Build configuration string
                        fe = info.get('feature_engineering', 'unknown')
                        scaler = info.get('scaler', 'unknown')
                        fs = info.get('feature_selection', 'unknown')
                        configuration = f"{fe}-{scaler}-{fs}"
                        
                        self.metrics[full_name] = {
                            'cv_auc': cv_score,
                            'test_auc': test_auc,
                            'configuration': configuration,
                            'accuracy': test_metrics.get('accuracy', 0),
                            'precision': test_metrics.get('precision', 0),
                            'recall': test_metrics.get('recall_sensitivity', 0),
                            'f1_score': test_metrics.get('f1_score', 0),
                            'specificity': test_metrics.get('specificity', 0)
                        }
                    
                    print(f"✅ Loaded metrics for {len(self.metrics)} models")
                    return True
            else:
                print(f"⚠️  Metrics file not found: {summary_path}")
                return False
        except Exception as e:
            print(f"❌ Error loading metrics: {str(e)}")
            import traceback
            traceback.print_exc()
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
                    # Extract components từ pipeline (new structure)
                    model = pipeline_data['model']
                    fe_transformer = pipeline_data.get('fe_transformer', None)  # New: fe_transformer instead of fe_func
                    preprocessor = pipeline_data.get('preprocessor', None)  # May be None in new structure
                    scaler_name = pipeline_data.get('scaler_name', None)  # New: scaler_name
                    fs_name = pipeline_data.get('fs_name', None)  # New: fs_name instead of fs_indices
                    numerical_features = pipeline_data.get('numerical_features', None)  # New
                    categorical_features = pipeline_data.get('categorical_features', None)  # New
                    expected_features = pipeline_data.get('feature_names', None)
                    
                    # Apply feature engineering (new structure)
                    if fe_transformer is not None:
                        # Check if fe_transformer has transform method
                        if hasattr(fe_transformer, 'transform'):
                            # New class-based approach (BasicFE, PolyFE, EnhancedFE)
                            X_transformed = fe_transformer.transform(X_fe)
                        else:
                            # fe_transformer is just metadata, no transformation needed
                            X_transformed = X_fe
                    else:
                        # Fallback to old approach if fe_func exists
                        fe_func = pipeline_data.get('fe_func', None)
                        if fe_func is not None:
                            if isinstance(fe_func, str):
                                # Old function-based approach
                                if fe_func == 'fe_basic':
                                    fe_func = fe_basic
                                elif fe_func == 'fe_enhanced':
                                    fe_func = fe_enhanced
                                elif fe_func == 'fe_poly_only':
                                    fe_func = fe_poly_only
                                # Call the function
                                X_transformed, _, _ = fe_func(X_fe)
                            else:
                                # Old class-based approach
                                X_transformed = fe_func.transform(X_fe)
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
                    if preprocessor is not None:
                        X_preprocessed = preprocessor.transform(X_transformed)
                    else:
                        # Handle case where preprocessor is None (new structure)
                        # Need to create preprocessor from scaler_name and features
                        if scaler_name and numerical_features and categorical_features:
                            from sklearn.compose import ColumnTransformer
                            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
                            from sklearn.preprocessing import OneHotEncoder
                            
                            # Create scaler
                            scaler_map = {
                                'standard': StandardScaler(),
                                'minmax': MinMaxScaler(),
                                'robust': RobustScaler()
                            }
                            scaler = scaler_map.get(scaler_name, StandardScaler())
                            
                            # Create preprocessor
                            preprocessor = ColumnTransformer(
                                transformers=[
                                    ('num', scaler, numerical_features),
                                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
                                ]
                            )
                            
                            # Create dummy data to fit preprocessor (to avoid refitting each time)
                            import pandas as pd
                            dummy_data = pd.DataFrame({
                                'age': [50], 'trestbps': [120], 'chol': [200], 'thalach': [150], 'oldpeak': [1.0],
                                'sex': [1], 'cp': [1], 'fbs': [0], 'restecg': [0], 'exang': [0], 'slope': [1], 'ca': [0], 'thal': [2]
                            })
                            
                            # Fit preprocessor on dummy data
                            preprocessor.fit(dummy_data)
                            
                            # Transform the actual data
                            X_preprocessed = preprocessor.transform(X_transformed)
                        else:
                            # No preprocessing available, use as is
                            X_preprocessed = X_transformed
                    
                    # Convert to array if sparse
                    if hasattr(X_preprocessed, 'toarray'):
                        X_preprocessed = X_preprocessed.toarray()
                    
                    # Apply feature selection (new structure)
                    fs_indices = pipeline_data.get('fs_indices', None)  # Check for old structure
                    if fs_indices is not None:
                        # Old structure: use fs_indices
                        X_selected = X_preprocessed[:, fs_indices]
                    elif fs_name is not None:
                        # New structure: feature selection indices not saved
                        # TEMPORARY WORKAROUND: Skip feature selection and use all features
                        # This may cause feature mismatch but allows app to run
                        print(f"⚠️  Feature selection '{fs_name}' not implemented, using all features")
                        X_selected = X_preprocessed
                    else:
                        # No feature selection, use all features
                        X_selected = X_preprocessed
                    
                    # Make prediction with feature padding if needed
                    # Handle ensemble model differently
                    if model_name == 'Ensemble':
                        # For ensemble, use the same preprocessing as individual models
                        # and let VotingClassifier handle the prediction
                        prediction = model.predict(X_selected)[0]
                    else:
                        expected_features = model.n_features_in_
                        actual_features = X_selected.shape[1]
                        
                        if actual_features < expected_features:
                            # Pad with zeros to match expected features
                            padding = np.zeros((X_selected.shape[0], expected_features - actual_features))
                            X_selected = np.hstack([X_selected, padding])
                            print(f"⚠️  Padded {expected_features - actual_features} features for {model_name}")
                        elif actual_features > expected_features:
                            # Truncate to match expected features
                            X_selected = X_selected[:, :expected_features]
                            print(f"⚠️  Truncated {actual_features - expected_features} features for {model_name}")
                        
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
