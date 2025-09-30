#!/usr/bin/env python3
"""
Experiment Manager - Quản lý thí nghiệm ML
- Logging cấu hình, tham số, kết quả
- So sánh kết quả giữa các lần chạy
- Reproducibility với seed cố định
"""

import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ExperimentManager:
    """Quản lý thí nghiệm ML với logging và reproducibility"""
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        """
        Khởi tạo Experiment Manager
        
        Args:
            experiment_name: Tên thí nghiệm
            base_dir: Thư mục lưu kết quả
        """
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.experiment_dir = os.path.join(base_dir, experiment_name)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Tạo thư mục thí nghiệm
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "results"), exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Lưu metadata thí nghiệm
        self.metadata = {
            "experiment_name": experiment_name,
            "timestamp": self.timestamp,
            "base_dir": base_dir
        }
        
    def _setup_logging(self):
        """Setup logging system"""
        log_file = os.path.join(self.experiment_dir, "logs", f"experiment_{self.timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 Bắt đầu thí nghiệm: {self.experiment_name}")
        
    def set_seed(self, seed: int = 42):
        """Đặt seed cố định để đảm bảo reproducibility"""
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        
        self.metadata["seed"] = seed
        self.logger.info(f"🌱 Đặt seed: {seed}")
        
    def log_config(self, config: Dict[str, Any]):
        """Log cấu hình thí nghiệm"""
        self.metadata["config"] = config
        
        # Lưu config vào file JSON
        config_file = os.path.join(self.experiment_dir, "config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"📝 Đã lưu cấu hình: {config_file}")
        
    def log_parameters(self, model_name: str, params: Dict[str, Any]):
        """Log tham số model"""
        if "model_parameters" not in self.metadata:
            self.metadata["model_parameters"] = {}
            
        self.metadata["model_parameters"][model_name] = params
        self.logger.info(f"⚙️  Tham số {model_name}: {params}")
        
    def log_results(self, model_name: str, results: Dict[str, float], 
                   train_time: float = None, test_predictions: np.ndarray = None):
        """Log kết quả thí nghiệm"""
        if "results" not in self.metadata:
            self.metadata["results"] = {}
            
        result_entry = {
            "metrics": results,
            "train_time": train_time,
            "timestamp": datetime.now().isoformat()
        }
        
        self.metadata["results"][model_name] = result_entry
        
        # Lưu predictions nếu có
        if test_predictions is not None:
            pred_file = os.path.join(self.experiment_dir, "results", f"{model_name}_predictions.npy")
            np.save(pred_file, test_predictions)
            
        self.logger.info(f"📊 Kết quả {model_name}: {results}")
        
    def save_model(self, model, model_name: str):
        """Lưu model đã train"""
        model_file = os.path.join(self.experiment_dir, "models", f"{model_name}.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
            
        self.logger.info(f"💾 Đã lưu model: {model_file}")
        
    def compare_experiments(self, other_experiment_dir: str = None):
        """So sánh với thí nghiệm khác"""
        if other_experiment_dir is None:
            # So sánh với thí nghiệm trước đó
            experiment_dirs = [d for d in os.listdir(self.base_dir) 
                             if os.path.isdir(os.path.join(self.base_dir, d))]
            if len(experiment_dirs) > 1:
                other_experiment_dir = os.path.join(self.base_dir, experiment_dirs[-2])
        
        if other_experiment_dir and os.path.exists(other_experiment_dir):
            # Load kết quả thí nghiệm khác
            other_config_file = os.path.join(other_experiment_dir, "config.json")
            if os.path.exists(other_config_file):
                with open(other_config_file, 'r') as f:
                    other_config = json.load(f)
                    
                self.logger.info(f"🔄 So sánh với thí nghiệm: {other_experiment_dir}")
                return self._create_comparison_report(other_config)
        
        return None
        
    def _create_comparison_report(self, other_config: Dict):
        """Tạo báo cáo so sánh"""
        comparison = {
            "current_experiment": self.metadata,
            "previous_experiment": other_config,
            "comparison_timestamp": datetime.now().isoformat()
        }
        
        comparison_file = os.path.join(self.experiment_dir, "comparison_report.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
            
        return comparison
        
    def finalize_experiment(self):
        """Hoàn tất thí nghiệm và lưu metadata"""
        self.metadata["status"] = "completed"
        self.metadata["end_time"] = datetime.now().isoformat()
        
        # Lưu metadata tổng hợp
        metadata_file = os.path.join(self.experiment_dir, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"✅ Hoàn tất thí nghiệm: {self.experiment_name}")
        self.logger.info(f"📁 Kết quả lưu tại: {self.experiment_dir}")
        
    def get_best_model(self) -> Optional[Dict]:
        """Lấy model có kết quả tốt nhất"""
        if "results" not in self.metadata:
            return None
            
        best_model = None
        best_score = -1
        
        for model_name, result in self.metadata["results"].items():
            # Sử dụng accuracy làm metric chính
            if "accuracy" in result["metrics"]:
                score = result["metrics"]["accuracy"]
                if score > best_score:
                    best_score = score
                    best_model = {
                        "name": model_name,
                        "score": score,
                        "metrics": result["metrics"]
                    }
                    
        return best_model

def create_enhanced_grid_search(estimator, param_grid: Dict, X, y, 
                              cv_splits: int = 5, scoring: str = 'accuracy',
                              experiment_manager: ExperimentManager = None):
    """GridSearchCV mở rộng với logging"""
    
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = datetime.now()
    grid_search.fit(X, y)
    end_time = datetime.now()
    
    train_time = (end_time - start_time).total_seconds()
    
    if experiment_manager:
        experiment_manager.log_parameters("grid_search", {
            "param_grid": param_grid,
            "cv_splits": cv_splits,
            "scoring": scoring
        })
        
        experiment_manager.log_results("grid_search", {
            "best_score": grid_search.best_score_,
            "best_params": grid_search.best_params_,
            "cv_scores": grid_search.cv_results_['mean_test_score'].tolist()
        }, train_time)
        
        experiment_manager.save_model(grid_search.best_estimator_, "best_model")
    
    return grid_search, train_time
