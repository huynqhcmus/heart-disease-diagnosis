# Heart Disease Diagnosis Using Ensemble Learning

**AIO2025 Research Project | VietAI Learning Team**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)](https://heart-disease-diagnosis-vietailearningteam.streamlit.app)

---

## Abstract

This study implements an ensemble learning approach for heart disease diagnosis using eight machine learning algorithms on the Cleveland Heart Disease dataset from UCI. The system achieves an average AUC of 0.92 across models, with comprehensive hyperparameter optimization and experiment tracking.

**Live Demo:** [https://heart-disease-diagnosis-vietai.streamlit.app](https://heart-disease-diagnosis-vietai.streamlit.app)

---

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Run application
./scripts/run_app.sh

# Access at http://localhost:8501
```

---

## Project Structure

```
heart-disease-diagnosis/
├── app/
│   └── streamlit_app.py          # Web interface
├── src/
│   ├── pipeline.py                # ML pipeline
│   ├── model_functions.py         # Feature engineering
│   └── utils/
│       └── app_utils.py           # Helper functions
├── scripts/
│   ├── experiment_manager.py      # Experiment tracking
│   ├── train_models.py            # Hyperparameter tuning
│   └── run_app.sh                 # Application launcher
├── data/
│   ├── raw/                       # Original data
│   └── processed/                 # Preprocessed splits
├── models/
│   └── saved_models/latest/       # Trained models
├── experiments/
│   ├── logs/                      # Training logs
│   ├── results/                   # Metrics & predictions
│   └── optimized_params/          # Best hyperparameters
└── notebooks/                     # Jupyter notebooks
```

---

## Methodology

### Dataset

**Source:** Cleveland Heart Disease Dataset (UCI Machine Learning Repository)  
**Samples:** 303 patients  
**Features:** 13 clinical attributes  
**Target:** Binary classification (0 = Healthy, 1 = Disease)

### Models Evaluated

Eight classification algorithms were systematically evaluated:

1. **Logistic Regression** - Linear baseline
2. **Random Forest** - Ensemble decision trees
3. **Support Vector Machine** - Kernel-based classifier
4. **Naive Bayes** - Probabilistic model
5. **K-Nearest Neighbors** - Instance-based learning
6. **Gradient Boosting** - Sequential ensemble
7. **Decision Tree** - Single tree classifier
8. **Voting Ensemble** - Meta-classifier

### Hyperparameter Optimization

- **Framework:** Optuna (Tree-structured Parzen Estimator)
- **Trials:** 100 per model
- **Validation:** 5-fold stratified cross-validation
- **Metric:** F1-score (macro average)

### Evaluation

- **Cross-validation AUC:** Performance during training
- **Test AUC:** Held-out test set performance
- **Majority Voting:** Final prediction from ensemble

---

## Results

| Model | CV AUC | Test AUC | Status |
|-------|--------|----------|--------|
| Logistic Regression | 0.9470 | 0.9470 | ✓ |
| Random Forest | 0.9394 | 0.9394 | ✓ |
| Support Vector Machine | 0.9351 | 0.9351 | ✓ |
| Naive Bayes | 0.9286 | 0.9286 | ✓ |
| K-Nearest Neighbors | 0.9221 | 0.9221 | ✓ |
| Gradient Boosting | 0.8546 | 0.9076 | ✓ |
| Voting Ensemble | 0.8890 | 0.9160 | ✓ |
| Decision Tree | 0.8561 | 0.8561 | ✓ |

**Average AUC:** 0.920  
**Best Model:** Logistic Regression (0.947)

---

## Application Features

### 1. Patient Diagnosis
- Interactive input form with clinical parameter validation
- Real-time prediction from 8 models
- Majority voting with confidence scores
- Risk assessment visualization

### 2. Model Analysis
- Comprehensive performance metrics
- Cross-validation vs. test set comparison
- Model configuration details

### 3. Feature Importance
- SHAP-style feature contribution analysis
- Model-specific importance rankings
- Clinical interpretation guides

### 4. Experiment Tracking
- Complete hyperparameter search history
- Reproducible experiment logs
- Performance comparison tools

### 5. History & Reports
- Patient prediction archive
- PDF report generation
- Export capabilities

---

## Installation

### Prerequisites

- Python 3.10+
- pip package manager

### Setup

```bash
# Clone repository
git clone https://github.com/huynqhcmus/heart-disease-diagnosis.git
cd heart-disease-diagnosis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Local Deployment

```bash
./scripts/run_app.sh
```

### Cloud Deployment

Application is deployed on Streamlit Cloud:
- **URL:** https://heart-disease-diagnosis-vietailearningteam.streamlit.app
- **Auto-deploy:** Triggered on Git push
- **Uptime:** 24/7 availability

See `docs/DEPLOYMENT.md` for details.

### Hyperparameter Tuning

```bash
python scripts/train_models.py
```

Results saved to `experiments/optimized_params/`

---

## Technical Details

### Dependencies

- **ML:** scikit-learn 1.7.2, XGBoost, LightGBM
- **UI:** Streamlit 1.25+
- **Optimization:** Optuna
- **Visualization:** Plotly
- **Utils:** pandas, numpy, joblib

### Reproducibility

- Fixed random seed (42) for all experiments
- Complete hyperparameter logging
- Versioned model artifacts

---

## Limitations & Disclaimers

⚠️ **For Educational/Research Purposes Only**

This system is NOT intended for clinical use. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.

**Known Limitations:**
- Small dataset size (n=303)
- Limited to Cleveland clinic population
- No external validation cohort
- Missing feature: temporal trends

---

## Team

**VietAI Learning Team - AIO2025**

- Dũng
- Anh
- Vinh
- Hằng
- Huy

---

## Acknowledgments

- **VietAI** for the AIO2025 Machine Learning course
- **UCI Machine Learning Repository** for the Cleveland Heart Disease dataset
- Open-source communities: scikit-learn, Streamlit, Optuna

---

## Citation

If you use this work, please cite:

```
VietAI Learning Team (2025). Heart Disease Diagnosis Using Ensemble Learning.
AIO2025 Research Project. https://github.com/huynqhcmus/heart-disease-diagnosis
```

---

## License

Educational use only. See individual package licenses for dependencies.

---

**Last Updated:** September 30, 2025  
**Version:** 1.0 - Production Ready# Trigger redeploy
