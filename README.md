# 🫀 Heart Disease Diagnosis - Ensemble Learning

**AIO2025 Project - Machine Learning for Medical Diagnosis**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25-red)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](https://github.com)

---

## 👥 VietAI Learning Team

**Team Members:** Dũng, Anh, Vinh, Hằng, Huy  
**Course:** AIO2025 - Advanced Machine Learning  
**Organization:** VietAI

---

## 🎉 PROJECT STATUS - HOÀN THÀNH ✅

**Ngày hoàn thành:** 30/09/2025

### 🎯 Kết quả chính:
- ✅ **8/8 models hoạt động** (LR, RF, KNN, DT, GB, NB, SVM, Ensemble)
- ✅ **App demo hoàn chỉnh** với giao diện hiện đại
- ✅ **Experiment Manager** để quản lý thí nghiệm
- ✅ **Hyperparameter Tuning** tự động với Optuna
- ✅ **PDF Reports & History** tracking

---

## 🚀 Quick Start (3 Bước)

### 1️⃣ Setup Environment
\`\`\`bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (nếu chưa có)
pip install -r requirements.txt
\`\`\`

### 2️⃣ Run App
\`\`\`bash
# Cách 1: Dùng script (recommended)
./run_app.sh

# Cách 2: Direct command
streamlit run improved_app.py
\`\`\`

### 3️⃣ Use App
1. Mở browser tại \`http://localhost:8501\`
2. Click **"🚀 Initialize Prediction System"**
3. Nhập thông tin bệnh nhân (hoặc chọn preset example)
4. Click **"🔮 Run Diagnosis (All Models)"**
5. Xem kết quả trong 5 tabs!

---

## 📊 Dataset

- **Nguồn**: Cleveland Heart Disease Dataset (UCI)
- **Samples**: ~300 patients  
- **Features**: 13 clinical features
- **Target**: Binary (0 = No Disease, 1 = Disease)

### Features (13)

**Numerical:** age, trestbps, chol, thalach, oldpeak  
**Categorical:** sex, cp, fbs, restecg, exang, slope, ca, thal

---

## 🤖 Models (8 Active)

| Model | Test AUC | Status |
|-------|----------|--------|
| Logistic Regression | 0.9470 | ✅ Active |
| Random Forest | 0.9394 | ✅ Active |
| SVM | 0.9351 | ✅ Active |
| Naive Bayes | 0.9286 | ✅ Active |
| K-Nearest Neighbors | 0.9221 | ✅ Active |
| Gradient Boosting | 0.9076 | ✅ Active |
| Ensemble (Voting) | 0.9160 | ✅ Active |
| Decision Tree | 0.8561 | ✅ Active |

---

## 📁 Project Structure

\`\`\`
heart-disease-diagnosis/
├── improved_app.py              # 🎯 Main Streamlit app
├── pipeline.py                  # Data & prediction pipeline
├── model_functions.py           # Feature engineering
├── app_utils.py                 # Utilities (PDF, charts)
├── experiment_manager.py        # Experiment tracking
├── hyperparameter_tuning.py     # Optuna optimization
├── run_app.sh                   # App launcher
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── STATUS.md                    # Detailed status
│
├── data/processed/              # Train/val/test splits
├── models/saved_models/latest/  # 8 model pipelines
├── experiments/                 # Logs & reports
├── notebooks/                   # Training notebooks
└── logs/                        # App logs
\`\`\`

---

## ✨ App Features (5 Tabs)

### 🏥 Tab 1: Patient Input & Prediction
- Enhanced input form với sliders & tooltips
- 4 Preset Examples (Normal, Low/Medium/High Risk)
- Majority Voting từ 8 models
- Individual predictions với confidence scores
- Bar charts & visualizations
- Save history & Export PDF

### 📊 Tab 2: Model Analysis
- Performance summary table (CV AUC, Test AUC)
- Model status indicators
- Comparison charts

### 🔬 Tab 3: Feature Importance
- Interactive feature importance charts
- Feature descriptions
- Top features highlighted

### 📈 Tab 4: Experiment Tracking
- View all experiments
- Filter by model/dataset
- Generate HTML reports
- Export to CSV

### 📝 Tab 5: History & Reports
- Prediction history với timestamps
- Statistics dashboard
- Download & clear history

---

## 🔧 Advanced Usage

### Hyperparameter Tuning
\`\`\`bash
python hyperparameter_tuning.py
# Takes 2-4 hours, saves to experiments/optimized_params/
\`\`\`

### Custom Predictions
\`\`\`python
from pipeline import pipeline
import pandas as pd

pipeline.load_models()
results, predictions = pipeline.predict(patient_data)
final_pred, votes, total = pipeline.get_majority_vote(predictions)
\`\`\`

---

## 🎯 Preset Patient Examples

1. **Normal Patient** - Healthy individual
2. **Low Risk** - Mild risk factors  
3. **Medium Risk** - Several risk factors
4. **High Risk** - High risk profile

---

## 🐛 Troubleshooting

**App Won't Start:**
\`\`\`bash
python --version  # Check 3.8+
pip install -r requirements.txt --upgrade
streamlit --version
\`\`\`

**Models Won't Load:**
\`\`\`bash
ls -la models/saved_models/latest/
# Should see 8 .pkl files + 1 .json
\`\`\`

**Clear Cache:**
\`\`\`bash
streamlit cache clear
pkill -f streamlit
./run_app.sh
\`\`\`

---

## ⚖️ License & Disclaimer

**Educational Use Only**

⚠️ **Medical Disclaimer:**  
This tool is NOT a substitute for professional medical advice.  
Always consult qualified healthcare professionals for diagnosis.  
Do not use for actual clinical decision-making.

---

## 🙏 Acknowledgments

- **VietAI** for the AIO2025 course
- **UCI ML Repository** for the dataset
- **scikit-learn, Streamlit, Optuna** communities
- Course instructors and team members

---

**🫀 Enjoy using the Heart Disease Diagnosis System!**

*Last Updated: 30/09/2025 | Version: 1.0 - Production Ready*  
*Team: Dũng, Anh, Vinh, Hằng, Huy | AIO2025 VietAI*
