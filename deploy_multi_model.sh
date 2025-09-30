#!/bin/bash

# Thư mục gốc của project
PROJECT_ROOT="/Users/huynguyen/AIO2025/Project/heart-disease-diagnosis"
LOGS_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOGS_DIR/multi_model_app.pid"
LOG_FILE="$LOGS_DIR/multi_model_app.log"
STREAMLIT_APP="$PROJECT_ROOT/multi_model_app.py"
PORT=8506

# Tạo thư mục logs nếu chưa có
mkdir -p "$LOGS_DIR"

echo "🚀 Deploy Multi-Model Heart Disease Prediction App..."

# Chạy Streamlit server dưới nền
echo "📊 Chạy Multi-Model Streamlit app dưới nền..."
nohup streamlit run "$STREAMLIT_APP" \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    &> "$LOG_FILE" &

# Lưu PID của Streamlit process
echo $! > "$PID_FILE"

echo "⏳ Đợi 3 giây để server khởi động..."
sleep 3

echo "🌐 Tạo đường hầm với Cloudflared..."
echo "📝 Logs được lưu tại: $LOG_FILE"

# Tạo đường hầm với Cloudflared
# Chạy Cloudflared ở foreground để hiển thị URL
./cloudflared tunnel --url http://localhost:"$PORT" --no-autoupdate
