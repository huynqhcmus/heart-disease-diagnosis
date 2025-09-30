"""
Script to calculate real performance metrics for all models
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pipeline import HeartDiseasePipeline

def calculate_real_metrics():
    """Calculate real performance metrics for all models"""
    
    # Initialize pipeline
    pipeline = HeartDiseasePipeline()
    if not pipeline.initialize():
        print("Failed to initialize pipeline")
        return None
    
    # Load test data
    try:
        # Load test data (assuming it exists)
        test_data_path = "data/processed/raw_test.csv"
        if os.path.exists(test_data_path):
            column_names = ['age', 'sex', 'cp', 'trestbpd', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            df = pd.read_csv(test_data_path, names=column_names)
            X_test = df[['age', 'sex', 'cp', 'trestbpd', 'chol', 'fbs', 'restecg', 
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
            y_test = df['target']
        else:
            print(f"Test data not found at {test_data_path}")
            return None
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None
    
    # Calculate metrics for each model
    results = []
    
    for model_name, model in pipeline.models.items():
        try:
            # Preprocess test data
            X_test_processed = pipeline.preprocessor.transform(X_test)
            
            # Make predictions
            y_pred = model.predict(X_test_processed)
            y_proba = model.predict_proba(X_test_processed)[:, 1]  # Probability of class 1
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            roc_auc = roc_auc_score(y_test, y_proba)
            
            results.append({
                'Model': model_name,
                'ROC-AUC': roc_auc,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            })
            
            print(f"✅ {model_name}: ROC-AUC={roc_auc:.4f}, Accuracy={accuracy:.4f}")
            
        except Exception as e:
            print(f"❌ Error calculating metrics for {model_name}: {e}")
            continue
    
    # Create DataFrame and sort by ROC-AUC
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
    results_df['Rank'] = range(1, len(results_df) + 1)
    results_df = results_df[['Rank', 'Model', 'ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1']]
    
    # Format numbers
    numeric_cols = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
    for col in numeric_cols:
        results_df[col] = results_df[col].apply(lambda x: f"{x:.4f}")
    
    # Save results
    results_df.to_csv('real_model_metrics.csv', index=False)
    print(f"\n📊 Real metrics calculated and saved to 'real_model_metrics.csv'")
    print("\n🏆 Model Performance Comparison (Real Metrics):")
    print(results_df.to_string(index=False))
    
    return results_df

if __name__ == "__main__":
    calculate_real_metrics()
