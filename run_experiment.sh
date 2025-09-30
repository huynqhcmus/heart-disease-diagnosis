#!/bin/bash

# Script chạy thí nghiệm với Experiment Management
echo "🚀 Bắt đầu thí nghiệm Heart Disease với Experiment Management..."

# Tạo thư mục experiments nếu chưa có
mkdir -p experiments

# Kiểm tra dependencies
echo "📦 Kiểm tra dependencies..."
python -c "import pandas, numpy, sklearn, xgboost" 2>/dev/null || {
    echo "❌ Thiếu dependencies. Đang cài đặt..."
    pip install -r requirements.txt
}

# Chạy enhanced training
echo "🔬 Chạy enhanced training với experiment management..."
python3 enhanced_training.py

echo "✅ Hoàn thành thí nghiệm!"
echo "📁 Kết quả được lưu trong thư mục experiments/"
echo "📊 Xem logs và kết quả chi tiết trong experiments/heart_disease_enhanced/"
