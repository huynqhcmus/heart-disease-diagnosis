import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Heart Disease Prediction - Multi Model",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1976d2;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .model-card {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        background: #f8f9fa;
        margin-top: 2rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_data():
    """Load multiple trained models and data"""
    try:
        # Load data
        data_path = "data/processed/raw_train.csv"
        if not os.path.exists(data_path):
            st.error(f"Data file not found: {data_path}")
            return None, None, None, None
        
        df = pd.read_csv(data_path)
        
        # Prepare features and target
        feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        X = df[feature_cols]
        y = df['target']
        
        # Create preprocessing pipeline
        numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Train multiple models
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'XGBoost': GradientBoostingClassifier(random_state=42, n_estimators=100),  # Using GB as XGBoost alternative
            'k-NN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        
        # Train models
        trained_models = {}
        for name, model in models.items():
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            pipeline.fit(X, y)
            trained_models[name] = pipeline
        
        # Create ensemble model
        ensemble_models = [
            ('dt', DecisionTreeClassifier(random_state=42, max_depth=5)),
            ('rf', RandomForestClassifier(random_state=42, n_estimators=50)),
            ('gb', GradientBoostingClassifier(random_state=42, n_estimators=50))
        ]
        
        ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
        ensemble_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', ensemble)
        ])
        ensemble_pipeline.fit(X, y)
        trained_models['Ensemble (Soft Voting)'] = ensemble_pipeline
        
        return trained_models, preprocessor, X, y
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def get_feature_importance(model, feature_names):
    """Extract feature importance from model"""
    try:
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importance = model.named_steps['classifier'].feature_importances_
            return dict(zip(feature_names, importance))
        elif hasattr(model.named_steps['classifier'], 'coef_'):
            # For linear models
            coef = model.named_steps['classifier'].coef_[0]
            return dict(zip(feature_names, np.abs(coef)))
        else:
            return None
    except:
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">Heart Disease Prediction - Multi Model</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered heart disease risk assessment using multiple machine learning models</p>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models and data..."):
        models, preprocessor, X, y = load_models_and_data()
    
    if models is None:
        st.error("Failed to load models. Please check your data files.")
        return
    
    # Sidebar for input
    st.sidebar.markdown("## Patient Parameters")
    
    with st.sidebar.form('input_form'):
        # Numerical inputs with sliders
        age = st.slider('Age (years)', 20, 80, 50)
        trestbps = st.slider('Resting Blood Pressure (mmHg)', 90, 200, 120)
        chol = st.slider('Serum Cholesterol (mg/dl)', 100, 600, 200)
        thalach = st.slider('Maximum Heart Rate', 70, 220, 150)
        oldpeak = st.slider('ST Depression', 0.0, 6.0, 1.0)
        
        # Categorical inputs with selectbox
        sex = st.selectbox('Gender', [('Female', 0), ('Male', 1)], format_func=lambda x: x[0])[1]
        cp = st.selectbox('Chest Pain Type', [
            ('Typical Angina', 1), ('Atypical Angina', 2), 
            ('Non-anginal Pain', 3), ('Asymptomatic', 4)
        ], format_func=lambda x: x[0])[1]
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
        restecg = st.selectbox('Resting ECG', [
            ('Normal', 0), ('ST-T Wave Abnormality', 1), ('Left Ventricular Hypertrophy', 2)
        ], format_func=lambda x: x[0])[1]
        exang = st.selectbox('Exercise Induced Angina', [('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
        slope = st.selectbox('ST Segment Slope', [
            ('Upsloping', 1), ('Flat', 2), ('Downsloping', 3)
        ], format_func=lambda x: x[0])[1]
        ca = st.selectbox('Number of Major Vessels', [('0', 0), ('1', 1), ('2', 2), ('3', 3)], format_func=lambda x: x[0])[1]
        thal = st.selectbox('Thalassemia', [
            ('Normal', 3), ('Fixed Defect', 6), ('Reversible Defect', 7)
        ], format_func=lambda x: x[0])[1]
        
        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)
    
    if submitted:
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
            'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
            'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
        })
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in models.items():
            try:
                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0]
                predictions[name] = pred
                probabilities[name] = prob
            except Exception as e:
                st.error(f"Error with {name}: {str(e)}")
                continue
        
        # Display results
        st.markdown("## 🎯 Model Predictions")
        
        # Create comparison chart
        model_names = list(predictions.keys())
        confidences = [max(probabilities[name]) for name in model_names]
        pred_labels = ['No Heart Disease' if predictions[name] == 0 else 'Heart Disease' for name in model_names]
        
        # Bar chart
        fig = px.bar(
            x=model_names, 
            y=confidences,
            title="Model Prediction Confidence",
            labels={'x': 'Model', 'y': 'Prediction Confidence'},
            color=confidences,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual model results
        st.markdown("### 📊 Individual Model Results")
        
        # Create two columns for model cards
        col1, col2 = st.columns(2)
        
        for i, (name, pred) in enumerate(predictions.items()):
            prob = probabilities[name]
            no_disease_prob = prob[0]
            disease_prob = prob[1]
            confidence = max(prob)
            
            # Determine which column to use
            col = col1 if i % 2 == 0 else col2
            
            with col:
                st.markdown(f"""
                <div class="model-card">
                    <h4>{name}</h4>
                    <p><strong>Prediction:</strong> {'✅ No Heart Disease' if pred == 0 else '⚠️ Heart Disease'}</p>
                    <p><strong>Confidence:</strong> {confidence:.1%}</p>
                    <p><strong>P(No disease):</strong> {no_disease_prob:.3f}</p>
                    <p><strong>P(Heart disease):</strong> {disease_prob:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Summary table
        st.markdown("### 📋 All Model Predictions")
        summary_data = []
        for name, pred in predictions.items():
            prob = probabilities[name]
            summary_data.append({
                'Model': name,
                'Prediction': 'No Heart Disease' if pred == 0 else 'Heart Disease',
                'Confidence': f"{max(prob):.1%}",
                'P(No disease)': f"{prob[0]:.3f}",
                'P(Heart disease)': f"{prob[1]:.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Feature importance (using the best model)
        best_model_name = max(predictions.keys(), key=lambda x: max(probabilities[x]))
        best_model = models[best_model_name]
        
        try:
            feature_names = preprocessor.get_feature_names_out()
            importance = get_feature_importance(best_model, feature_names)
            
            if importance:
                st.markdown("### 🔍 Feature Importance")
                st.markdown(f"*Based on {best_model_name} model*")
                
                # Sort features by importance
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                
                # Create feature importance chart
                feature_names_clean = [name.split('__')[-1] for name, _ in sorted_features[:10]]
                importance_values = [imp for _, imp in sorted_features[:10]]
                
                fig_importance = px.bar(
                    x=importance_values,
                    y=feature_names_clean,
                    orientation='h',
                    title="Top 10 Most Important Features",
                    labels={'x': 'Importance', 'y': 'Feature'}
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
                
        except Exception as e:
            st.warning(f"Could not display feature importance: {str(e)}")
        
        # Prediction history (simple session state)
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        
        # Add current prediction to history
        history_entry = {
            'timestamp': pd.Timestamp.now(),
            'input': input_data.iloc[0].to_dict(),
            'predictions': predictions.copy(),
            'probabilities': probabilities.copy()
        }
        st.session_state.prediction_history.append(history_entry)
        
        # Display recent predictions
        if len(st.session_state.prediction_history) > 1:
            st.markdown("### 📈 Recent Predictions")
            recent_df = pd.DataFrame([
                {
                    'Time': entry['timestamp'].strftime('%H:%M:%S'),
                    'Age': entry['input']['age'],
                    'Gender': 'Male' if entry['input']['sex'] == 1 else 'Female',
                    'Chest Pain': entry['input']['cp'],
                    'Majority Prediction': 'No Heart Disease' if sum(entry['predictions'].values()) < len(entry['predictions'])/2 else 'Heart Disease'
                }
                for entry in st.session_state.prediction_history[-5:]
            ])
            st.dataframe(recent_df, use_container_width=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>Created by AIViet Learning Team by AI Viet Nam (AIO2025)</strong></p>
        <p>Multi-Model Heart Disease Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
