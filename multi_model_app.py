import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pipeline import pipeline
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide"
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

# Initialize pipeline
@st.cache_resource
def initialize_pipeline():
    """Initialize the prediction pipeline"""
    return pipeline.initialize()

# --- Sidebar for User Input ---
def get_user_input():
    st.sidebar.header("Patient Data Input")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        age = st.number_input('Age', 1, 120, 50, help="Patient's age in years.")
        sex_option = st.selectbox('Sex', ['Male', 'Female'], help="Patient's sex.")
        sex = 1 if sex_option == 'Male' else 0
        cp = st.selectbox('Chest Pain Type (CP)', [
            ('Typical Angina', 1), ('Atypical Angina', 2), ('Non-anginal Pain', 3), ('Asymptomatic', 4)
        ], format_func=lambda x: x[0], help="Type of chest pain experienced.")
        trestbpd = st.number_input('Resting Blood Pressure (trestbpd)', 50, 250, 120, help="In mm Hg.")
        chol = st.number_input('Serum Cholestoral (chol)', 100, 600, 200, help="In mg/dl.")
        fbs_option = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', ['True', 'False'])
        fbs = 1 if fbs_option == 'True' else 0
    with col2:
        restecg = st.selectbox('Resting ECG Results (restecg)', [
            ('Normal', 0), ('ST-T wave abnormality', 1), ('Left ventricular hypertrophy', 2)
        ], format_func=lambda x: x[0])
        thalach = st.number_input('Max Heart Rate Achieved (thalach)', 50, 220, 150)
        exang_option = st.selectbox('Exercise Induced Angina (exang)', ['Yes', 'No'])
        exang = 1 if exang_option == 'Yes' else 0
        oldpeak = st.number_input('ST depression (oldpeak)', 0.0, 10.0, 1.0, 0.1)
        slope = st.selectbox('Slope of ST segment', [
            ('Upsloping', 1), ('Flat', 2), ('Downsloping', 3)
        ], format_func=lambda x: x[0])
        ca = st.selectbox('Number of Major Vessels (ca)', [
            ('0', 0), ('1', 1), ('2', 2), ('3', 3)
        ], format_func=lambda x: x[0])
        thal = st.selectbox('Thalassemia (thal)', [
            ('Normal', 3), ('Fixed Defect', 6), ('Reversible Defect', 7)
        ], format_func=lambda x: x[0])

    user_data = {
        'age': age, 'sex': sex, 'cp': cp[1], 'trestbpd': trestbpd, 'chol': chol,
        'fbs': fbs, 'restecg': restecg[1], 'thalach': thalach, 'exang': exang, 
        'oldpeak': oldpeak, 'slope': slope[1], 'ca': ca[1], 'thal': thal[1]
    }
    feature_order = ['age', 'sex', 'cp', 'trestbpd', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    return pd.DataFrame([user_data])[feature_order]

# --- Main Application ---
# Initialize pipeline (moved to button click to avoid reset)

st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("This application uses seven ML models to predict the likelihood of heart disease.")

# --- NEW: Create a performance summary table ---
if hasattr(pipeline, 'metrics') and pipeline.metrics:
    performance_data = []
    for name, report in pipeline.metrics.items():
        performance_data.append({
            "Model": name,
            "Test Accuracy": f"{report['accuracy']:.4f}",
            "Validation Accuracy": f"{report['validation_accuracy']:.4f}"
        })
    performance_df = pd.DataFrame(performance_data).set_index('Model')

    with st.expander("View Model Performance"):
        st.dataframe(performance_df)

user_input_df = get_user_input()
st.subheader("Patient's Input Data")
st.dataframe(user_input_df)

if st.button("Run All Models & Predict", type="primary", use_container_width=True):
    try:
        # Initialize pipeline if not already done
        if not pipeline.is_fitted:
            with st.spinner("Initializing pipeline..."):
                if not pipeline.initialize():
                    st.error("Failed to initialize prediction pipeline")
                    st.stop()
        
        # Use pipeline to make predictions
        all_results, predictions = pipeline.predict(user_input_df)
        
        if not all_results:
            st.error("No predictions could be made. Please check your input data.")
            st.stop()
            
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        st.stop()
        
    st.write("---")
    
    # --- NEW: Add a final verdict based on majority vote ---
    st.subheader("Final Verdict (Majority Vote)")
    final_prediction, vote_count, total_models = pipeline.get_majority_vote(predictions)
    
    # Update labels for consistency
    final_prediction = "Heart Disease" if final_prediction == "High Risk" else "No Heart Disease"
    
    if final_prediction == "Heart Disease":
        st.error(f"**Heart Disease Detected** ({vote_count} out of {total_models} models agree).", icon="❗")
    else:
        st.success(f"**No Heart Disease Detected** ({vote_count} out of {total_models} models agree).", icon="✅")

    st.write("---")
    st.subheader("Individual Model Predictions")

    num_models = len(all_results)
    
    if num_models > 0:
        # Create columns with equal width
        cols = st.columns(num_models)
        
        for i, result in enumerate(all_results):
            # Update prediction labels
            prediction_label = "Heart Disease" if result['Prediction'] == "High Risk" else "No Heart Disease"
            
            with cols[i]:
                # Use container for better alignment
                with st.container():
                    st.markdown(f"**{result['Model']}**")
                    st.markdown("---")
                    
                    # Prediction with consistent styling
                    if prediction_label == "Heart Disease":
                        st.markdown(f"<div style='text-align: center; color: red; font-weight: bold; margin: 10px 0;'>{prediction_label}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align: center; color: green; font-weight: bold; margin: 10px 0;'>{prediction_label}</div>", unsafe_allow_html=True)
                    
                    # Confidence with consistent styling
                    st.markdown(f"<div style='text-align: center; font-size: 1.2em; font-weight: bold; margin: 10px 0;'>{result['Confidence']:.2%}</div>", unsafe_allow_html=True)
    else:
        st.error("No models were able to make predictions. Please check the model files.")
            
    st.write("---")
    st.subheader("Results Summary")

    if len(all_results) > 0:
        summary_df = pd.DataFrame(all_results)
        summary_df_display = summary_df.copy()
        
        # Update prediction labels in dataframe
        summary_df['Prediction'] = summary_df['Prediction'].apply(
            lambda x: "Heart Disease" if x == "High Risk" else "No Heart Disease"
        )
        summary_df_display['Prediction'] = summary_df_display['Prediction'].apply(
            lambda x: "Heart Disease" if x == "High Risk" else "No Heart Disease"
        )
        summary_df_display['Confidence'] = summary_df_display['Confidence'].apply(lambda x: f"{x:.2%}")
        
        # Create enhanced bar chart with plotly.graph_objects for more control
        fig = go.Figure()
        
        # Separate data by prediction type
        for _, row in summary_df.iterrows():
            color = '#d32f2f' if row['Prediction'] == 'Heart Disease' else '#388e3c'  # Red for Heart Disease, Green for No Heart Disease
            
            # Add bar with text inside
            fig.add_trace(go.Bar(
                x=[row['Model']],
                y=[row['Confidence']],
                name=row['Prediction'],
                marker_color=color,
                text=f"<b>{row['Prediction']}</b>",
                textposition='inside',
                textfont=dict(color='white', size=14),
                hovertemplate=f"<b>{row['Model']}</b><br>" +
                             f"Prediction: {row['Prediction']}<br>" +
                             f"Confidence: {row['Confidence']:.1%}<br>" +
                             "<extra></extra>",
                showlegend=False
            ))
        
        # Update layout for professional appearance with white background
        fig.update_layout(
            title={
                'text': 'Model Predictions',
                'font': {'size': 20, 'color': '#333'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis=dict(
                title='Model',
                tickangle=-45,
                tickfont=dict(size=12),
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                title='Prediction Confidence',
                tickformat='.0%',
                range=[0, 1.05],
                tickfont=dict(size=12),
                gridcolor='rgba(128,128,128,0.2)',
                dtick=0.2
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500,
            margin=dict(t=80, b=80, l=80, r=80),
            bargap=0.2,
            showlegend=False,
            hovermode='x unified'
        )
        
        # Add grid lines for better readability
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add a legend manually below the chart
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; margin-top: -20px;'>
                <span style='color: #d32f2f; font-size: 16px;'>● Heart Disease</span>
                &nbsp;&nbsp;&nbsp;&nbsp;
                <span style='color: #388e3c; font-size: 16px;'>● No Heart Disease</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Table below (bottom)
        st.markdown("##### Detailed Comparison Table")
        st.dataframe(summary_df_display, use_container_width=True)
    else:
        st.warning("No results to display.")
