#!/usr/bin/env python3
"""
Script phân tích kết quả thí nghiệm
- So sánh kết quả giữa các lần chạy
- Tạo báo cáo tổng hợp
- Visualize kết quả
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def load_experiment_results(experiment_dir: str):
    """Load kết quả từ thí nghiệm"""
    metadata_file = os.path.join(experiment_dir, "metadata.json")
    
    if not os.path.exists(metadata_file):
        print(f"❌ Không tìm thấy metadata.json trong {experiment_dir}")
        return None
        
    with open(metadata_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_results_dataframe(experiments_data):
    """Tạo DataFrame từ kết quả thí nghiệm"""
    results = []
    
    for exp_name, exp_data in experiments_data.items():
        if "results" in exp_data:
            for model_name, result in exp_data["results"].items():
                row = {
                    "experiment": exp_name,
                    "model": model_name,
                    "timestamp": result.get("timestamp", ""),
                    "train_time": result.get("train_time", 0)
                }
                
                # Thêm metrics
                if "metrics" in result:
                    row.update(result["metrics"])
                    
                results.append(row)
    
    return pd.DataFrame(results)

def plot_model_comparison(df):
    """Vẽ biểu đồ so sánh models"""
    if df.empty:
        print("❌ Không có dữ liệu để vẽ biểu đồ")
        return
        
    # Tạo subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    # Accuracy comparison
    if 'accuracy' in df.columns:
        sns.barplot(data=df, x='model', y='accuracy', ax=axes[0,0])
        axes[0,0].set_title('Accuracy Comparison')
        axes[0,0].tick_params(axis='x', rotation=45)
    
    # F1 Score comparison
    if 'f1' in df.columns:
        sns.barplot(data=df, x='model', y='f1', ax=axes[0,1])
        axes[0,1].set_title('F1 Score Comparison')
        axes[0,1].tick_params(axis='x', rotation=45)
    
    # ROC AUC comparison
    if 'roc_auc' in df.columns:
        sns.barplot(data=df, x='model', y='roc_auc', ax=axes[1,0])
        axes[1,0].set_title('ROC AUC Comparison')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # Training time comparison
    if 'train_time' in df.columns:
        sns.barplot(data=df, x='model', y='train_time', ax=axes[1,1])
        axes[1,1].set_title('Training Time Comparison')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('experiment_results_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(df):
    """Tạo báo cáo tổng hợp"""
    if df.empty:
        return "❌ Không có dữ liệu để tạo báo cáo"
    
    report = []
    report.append("📊 BÁO CÁO TỔNG HỢP THÍ NGHIỆM")
    report.append("=" * 50)
    
    # Thống kê tổng quan
    report.append(f"\n📈 Tổng số thí nghiệm: {df['experiment'].nunique()}")
    report.append(f"🤖 Tổng số models: {df['model'].nunique()}")
    
    # Model tốt nhất theo từng metric
    metrics = ['accuracy', 'f1', 'roc_auc']
    for metric in metrics:
        if metric in df.columns:
            best_idx = df[metric].idxmax()
            best_model = df.loc[best_idx, 'model']
            best_score = df.loc[best_idx, metric]
            report.append(f"\n🏆 {metric.upper()} tốt nhất: {best_model} ({best_score:.4f})")
    
    # Thống kê training time
    if 'train_time' in df.columns:
        avg_time = df['train_time'].mean()
        total_time = df['train_time'].sum()
        report.append(f"\n⏱️  Thời gian training trung bình: {avg_time:.2f}s")
        report.append(f"⏱️  Tổng thời gian training: {total_time:.2f}s")
    
    # Top 5 models theo accuracy
    if 'accuracy' in df.columns:
        top_models = df.nlargest(5, 'accuracy')[['model', 'accuracy']]
        report.append(f"\n🥇 TOP 5 MODELS THEO ACCURACY:")
        for idx, row in top_models.iterrows():
            report.append(f"   {row['model']}: {row['accuracy']:.4f}")
    
    return "\n".join(report)

def main():
    """Main function"""
    print("🔍 Phân tích kết quả thí nghiệm...")
    
    # Tìm tất cả thí nghiệm
    experiments_dir = "experiments"
    if not os.path.exists(experiments_dir):
        print(f"❌ Không tìm thấy thư mục {experiments_dir}")
        return
    
    experiments_data = {}
    experiment_dirs = [d for d in os.listdir(experiments_dir) 
                      if os.path.isdir(os.path.join(experiments_dir, d))]
    
    if not experiment_dirs:
        print("❌ Không tìm thấy thí nghiệm nào")
        return
    
    # Load kết quả từ tất cả thí nghiệm
    for exp_dir in experiment_dirs:
        exp_path = os.path.join(experiments_dir, exp_dir)
        exp_data = load_experiment_results(exp_path)
        if exp_data:
            experiments_data[exp_dir] = exp_data
    
    if not experiments_data:
        print("❌ Không load được dữ liệu thí nghiệm nào")
        return
    
    # Tạo DataFrame
    df = create_results_dataframe(experiments_data)
    
    if df.empty:
        print("❌ Không có kết quả để phân tích")
        return
    
    print(f"✅ Đã load {len(df)} kết quả từ {len(experiments_data)} thí nghiệm")
    
    # Tạo báo cáo
    report = generate_summary_report(df)
    print("\n" + report)
    
    # Lưu báo cáo
    with open("./docs/experiment_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    # Vẽ biểu đồ
    try:
        plot_model_comparison(df)
        print("\n📊 Đã tạo biểu đồ so sánh: experiment_results_comparison.png")
    except Exception as e:
        print(f"⚠️  Không thể tạo biểu đồ: {e}")
    
    # Lưu DataFrame
    df.to_csv("./docs/experiment_results.csv", index=False)
    print("💾 Đã lưu kết quả chi tiết: experiment_results.csv")
    
    print("\n🎉 Hoàn thành phân tích thí nghiệm!")

if __name__ == "__main__":
    main()
