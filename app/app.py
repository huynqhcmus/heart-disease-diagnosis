import streamlit as st
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import base64
from io import BytesIO

st.set_page_config(
    page_title='Heart Disease Prediction', 
    layout='wide',
    page_icon='❤️'
)
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
COLUMNS = ['age','sex','cp','trestbps','chol','fbs','restecg',
           'thalach','exang','oldpeak','slope','ca','thal','target']
NUMERIC = ['age','trestbps','chol','thalach','oldpeak']
CATEGORICAL = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

@st.cache_resource
def find_best_model():
    df = pd.read_csv(URL, header=None, names=COLUMNS, na_values='?')
    df = df.dropna()
    df['target'] = (df['target'] > 0).astype(int)

    X = df[NUMERIC + CATEGORICAL]
    y = df['target']

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, NUMERIC),
        ('cat', cat_pipe, CATEGORICAL)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    param_grid = {'classifier__max_depth': range(3, 11)}

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search

model = find_best_model()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        color: #666;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #eee;
    }
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header with logo
st.markdown('<h1 class="main-header">Heart Disease Prediction App</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered heart disease risk assessment using machine learning</p>', unsafe_allow_html=True)

# Model info
st.sidebar.markdown("## Model Information")
st.sidebar.success(f"Optimal model depth: {model.best_params_['classifier__max_depth']}")
st.sidebar.info("Model: Decision Tree Classifier")
st.sidebar.info("Dataset: Cleveland Heart Disease")

with st.sidebar.form('input_form'):
    st.markdown("### Patient Parameters")
    age = st.slider('Age', 20, 80, 50)
    sex = st.selectbox('Gender', [('Male', 1), ('Female', 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox('Chest Pain Type', [
        ('Typical Angina', 1),
        ('Atypical Angina', 2),
        ('Non-anginal Pain', 3),
        ('Asymptomatic', 4)
    ], format_func=lambda x: x[0])[1]
    trestbps = st.slider('Resting Blood Pressure (mmHg)', 90, 200, 120)
    chol = st.slider('Serum Cholesterol (mg/dl)', 120, 570, 240)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [('False', 0), ('True', 1)], format_func=lambda x: x[0])[1]
    restecg = st.selectbox('Resting ECG', [
        ('Normal', 0),
        ('ST-T Abnormality', 1),
        ('LV Hypertrophy', 2)
    ], format_func=lambda x: x[0])[1]
    thalach = st.slider('Max Heart Rate Achieved', 70, 210, 150)
    exang = st.selectbox('Exercise-induced Angina', [('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
    oldpeak = st.slider('ST Depression', 0.0, 6.2, 1.0, step=0.1)
    slope = st.selectbox('Slope of Peak ST', [
        ('Upsloping', 1),
        ('Flat', 2),
        ('Downsloping', 3)
    ], format_func=lambda x: x[0])[1]
    ca = st.slider('Major Vessels colored by Fluoroscopy', 0, 3, 0)
    thal = st.selectbox('Thalassemia', [
        ('Normal', 3),
        ('Fixed Defect', 6),
        ('Reversible Defect', 7)
    ], format_func=lambda x: x[0])[1]
    submitted = st.form_submit_button('Predict', use_container_width=True)

if submitted:
    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }])
    proba = model.predict_proba(input_data)[0]
    prob_disease = proba[1]

    # Prediction Results
    st.markdown("## Prediction Results")
    
    # Create two columns for results
    col1, col2 = st.columns(2)
    
    with col1:
        if prob_disease > 0.5:
            st.markdown(f'''
            <div class="prediction-card">
                <h2>⚠️ Disease Likely</h2>
                <h1>{prob_disease:.1%}</h1>
                <p>Recommendation: Consult a healthcare professional</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="prediction-card">
                <h2>✅ Disease Unlikely</h2>
                <h1>{(1-prob_disease):.1%}</h1>
                <p>Continue maintaining a healthy lifestyle</p>
            </div>
            ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Probability Breakdown")
        st.metric("Disease Probability", f"{prob_disease:.1%}")
        st.metric("No Disease Probability", f"{(1-prob_disease):.1%}")
        confidence = max(prob_disease, 1-prob_disease)
        st.metric("Confidence Level", f"{confidence:.1%}")

    # Decision Tree Visualization
    st.markdown("## Model Decision Tree Visualization")
    st.write("This chart shows the decision rules of the best model found via Cross-Validation.")

    best_pipeline = model.best_estimator_
    preprocessor = best_pipeline.named_steps['preprocessor']
    classifier = best_pipeline.named_steps['classifier']

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
        fig, ax = plt.subplots(figsize=(25, 12))
        plot_tree(
            classifier,
            feature_names=feature_names,
            class_names=['No Disease', 'Disease'],
            filled=True,
            rounded=True,
            ax=ax,
            fontsize=10
        )
        st.pyplot(fig)
    except Exception as e:
        st.warning(f'Could not display the tree visualization. Error: {e}')

# Footer with logo and team info
st.markdown("---")

# Load and display logo
try:
    logo_path = "assets/AIVietLT-logo.png"
    with open(logo_path, "rb") as f:
        logo_data = f.read()
        logo_base64 = base64.b64encode(logo_data).decode()
    
    st.markdown(f'''
    <div class="logo-container">
        <img src="data:image/png;base64,{logo_base64}" width="100" height="100">
    </div>
    ''', unsafe_allow_html=True)
except:
    pass

st.markdown("""
<div class="footer">
    <p><strong>Created by AIViet Learning Team by AI Viet Nam AIO2025)</strong></p>
</div>
""", unsafe_allow_html=True)
