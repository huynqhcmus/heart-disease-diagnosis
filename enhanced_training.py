#!/usr/bin/env python3
"""
Enhanced Training Script với Experiment Management
- Sử dụng ExperimentManager để quản lý thí nghiệm
- Logging đầy đủ cấu hình, tham số, kết quả
- GridSearchCV mở rộng
- Reproducibility với seed cố định
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append('src')

from experiment_manager import ExperimentManager, create_enhanced_grid_search
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_data(filepath: str):
    """Load và chuẩn bị data"""
    df = pd.read_csv(filepath)
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

def evaluate_model(model, X_test, y_test):
    """Đánh giá model với nhiều metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    return metrics, y_pred

def main():
    """Main function với experiment management"""
    
    # Khởi tạo Experiment Manager
    exp_manager = ExperimentManager(
        experiment_name="heart_disease_enhanced",
        base_dir="experiments"
    )
    
    # Đặt seed cố định
    exp_manager.set_seed(42)
    
    # Cấu hình thí nghiệm
    config = {
        "datasets": {
            "raw": "data/processed/raw_train.csv",
            "fe": "data/processed/fe_train.csv",
            "dt": "data/processed/dt_train.csv",
            "fe_dt": "data/processed/fe_dt_train.csv"
        },
        "test_data": "data/processed/raw_test.csv",
        "models": {
            "RandomForest": {
                "estimator_name": "RandomForestClassifier",
                "param_grid": {
                    "n_estimators": [50, 100, 200, 300],
                    "max_depth": [3, 5, 7, 10],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "AdaBoost": {
                "estimator_name": "AdaBoostClassifier",
                "param_grid": {
                    "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.5, 1.0]
                }
            },
            "GradientBoosting": {
                "estimator_name": "GradientBoostingClassifier",
                "param_grid": {
                    "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.5],
                    "max_depth": [3, 5, 7]
                }
            },
            "XGBoost": {
                "estimator_name": "XGBClassifier",
                "param_grid": {
                    "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.5],
                    "max_depth": [3, 5, 7]
                }
            }
        },
        "cv_splits": 5,
        "scoring": "accuracy"
    }
    
    # Log cấu hình
    exp_manager.log_config(config)
    
    # Load test data
    X_test, y_test = load_data(config["test_data"])
    exp_manager.logger.info(f"📊 Test data shape: {X_test.shape}")
    
    # Train trên từng dataset
    for dataset_name, train_file in config["datasets"].items():
        exp_manager.logger.info(f"\n🔄 Training trên dataset: {dataset_name}")
        
        # Load training data
        X_train, y_train = load_data(train_file)
        exp_manager.logger.info(f"   Training data shape: {X_train.shape}")
        
        # Train từng model
        for model_name, model_config in config["models"].items():
            exp_manager.logger.info(f"\n   🤖 Training {model_name}...")
            
            try:
                # Tạo estimator dựa trên tên
                estimator_name = model_config["estimator_name"]
                if estimator_name == "RandomForestClassifier":
                    estimator = RandomForestClassifier(random_state=42)
                elif estimator_name == "AdaBoostClassifier":
                    estimator = AdaBoostClassifier(
                        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                        random_state=42
                    )
                elif estimator_name == "GradientBoostingClassifier":
                    estimator = GradientBoostingClassifier(random_state=42)
                elif estimator_name == "XGBClassifier":
                    estimator = XGBClassifier(random_state=42, eval_metric='logloss')
                else:
                    raise ValueError(f"Unknown estimator: {estimator_name}")
                
                # GridSearchCV với logging
                grid_search, train_time = create_enhanced_grid_search(
                    estimator=estimator,
                    param_grid=model_config["param_grid"],
                    X=X_train,
                    y=y_train,
                    cv_splits=config["cv_splits"],
                    scoring=config["scoring"],
                    experiment_manager=exp_manager
                )
                
                # Đánh giá trên test set
                metrics, y_pred = evaluate_model(grid_search.best_estimator_, X_test, y_test)
                
                # Log kết quả
                exp_manager.log_results(
                    model_name=f"{dataset_name}_{model_name}",
                    results=metrics,
                    train_time=train_time,
                    test_predictions=y_pred
                )
                
                # Lưu model
                exp_manager.save_model(
                    grid_search.best_estimator_, 
                    f"{dataset_name}_{model_name}"
                )
                
                exp_manager.logger.info(f"   ✅ {model_name} - Accuracy: {metrics['accuracy']:.4f}")
                
            except Exception as e:
                exp_manager.logger.error(f"   ❌ Lỗi khi train {model_name}: {str(e)}")
    
    # So sánh với thí nghiệm trước
    comparison = exp_manager.compare_experiments()
    if comparison:
        exp_manager.logger.info("📈 Đã tạo báo cáo so sánh với thí nghiệm trước")
    
    # Lấy model tốt nhất
    best_model = exp_manager.get_best_model()
    if best_model:
        exp_manager.logger.info(f"🏆 Model tốt nhất: {best_model['name']} (Score: {best_model['score']:.4f})")
    
    # Hoàn tất thí nghiệm
    exp_manager.finalize_experiment()
    
    print(f"\n🎉 Hoàn thành thí nghiệm!")
    print(f"📁 Kết quả lưu tại: {exp_manager.experiment_dir}")
    print(f"📊 Model tốt nhất: {best_model['name'] if best_model else 'Không có'}")

if __name__ == "__main__":
    main()
