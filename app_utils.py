"""
Utility Functions cho Heart Disease Diagnosis App
Các hàm hỗ trợ cho Streamlit app
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
from pathlib import Path
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.platypus import Image as RLImage


class PatientHistoryManager:
    """Quản lý lịch sử dự đoán của bệnh nhân"""
    
    def __init__(self, history_file="data/patient_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load_history()
    
    def _load_history(self):
        """Load lịch sử từ file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_history(self):
        """Save lịch sử vào file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_prediction(self, patient_id, patient_data, predictions, final_verdict):
        """
        Thêm một prediction vào history
        
        Args:
            patient_id (str): ID bệnh nhân
            patient_data (dict): Thông tin bệnh nhân
            predictions (list): Danh sách predictions từ các models
            final_verdict (str): Kết luận cuối cùng
        """
        record = {
            'patient_id': patient_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'patient_data': patient_data,
            'predictions': predictions,
            'final_verdict': final_verdict
        }
        
        self.history.append(record)
        self._save_history()
        return len(self.history) - 1
    
    def get_history(self, patient_id=None):
        """Lấy lịch sử (tất cả hoặc theo patient_id)"""
        if patient_id:
            return [h for h in self.history if h.get('patient_id') == patient_id]
        return self.history
    
    def get_statistics(self):
        """Tính toán thống kê từ history"""
        if not self.history:
            return {}
        
        df = pd.DataFrame(self.history)
        stats = {
            'total_predictions': len(df),
            'unique_patients': len(df['patient_id'].unique()) if 'patient_id' in df.columns else 0,
            'heart_disease_count': sum(1 for h in self.history if 'Heart Disease' in h.get('final_verdict', '')),
            'no_disease_count': sum(1 for h in self.history if 'No Heart Disease' in h.get('final_verdict', ''))
        }
        
        return stats


def create_patient_report_pdf(patient_data, predictions, final_verdict, output_file="patient_report.pdf"):
    """
    Tạo PDF report cho bệnh nhân
    
    Args:
        patient_data (dict): Thông tin bệnh nhân
        predictions (list): Danh sách predictions
        final_verdict (str): Kết luận cuối cùng
        output_file (str): Tên file output
    
    Returns:
        BytesIO: PDF file in memory
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1976d2'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#424242'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("🫀 Heart Disease Diagnosis Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"<b>Generated:</b> {timestamp}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Patient Information
    story.append(Paragraph("Patient Information", heading_style))
    patient_table_data = [['Feature', 'Value']]
    
    feature_names = {
        'age': 'Age (years)',
        'sex': 'Sex',
        'cp': 'Chest Pain Type',
        'trestbps': 'Resting Blood Pressure (mm Hg)',
        'chol': 'Cholesterol (mg/dl)',
        'fbs': 'Fasting Blood Sugar > 120 mg/dl',
        'restecg': 'Resting ECG',
        'thalach': 'Max Heart Rate',
        'exang': 'Exercise Induced Angina',
        'oldpeak': 'ST Depression',
        'slope': 'Slope of ST Segment',
        'ca': 'Number of Major Vessels',
        'thal': 'Thalassemia'
    }
    
    for key, value in patient_data.items():
        display_name = feature_names.get(key, key)
        patient_table_data.append([display_name, str(value)])
    
    patient_table = Table(patient_table_data, colWidths=[3*inch, 2*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Final Verdict
    story.append(Paragraph("Diagnosis Result", heading_style))
    verdict_color = colors.HexColor('#d32f2f') if 'Heart Disease' in final_verdict else colors.HexColor('#388e3c')
    verdict_text = f'<font color="{verdict_color.hexval()}"><b>{final_verdict}</b></font>'
    story.append(Paragraph(verdict_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Model Predictions
    story.append(Paragraph("Individual Model Predictions", heading_style))
    pred_table_data = [['Model', 'Prediction', 'Confidence']]
    
    for pred in predictions:
        pred_table_data.append([
            pred['Model'],
            pred['Prediction'],
            f"{pred['Confidence']:.2%}"
        ])
    
    pred_table = Table(pred_table_data, colWidths=[2*inch, 2*inch, 1.5*inch])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(pred_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Recommendations
    story.append(Paragraph("Recommendations", heading_style))
    if 'Heart Disease' in final_verdict:
        recommendations = """
        • Consult with a cardiologist immediately
        • Monitor blood pressure and cholesterol levels regularly
        • Adopt a heart-healthy diet (low in saturated fats and sodium)
        • Engage in regular physical activity as recommended by your doctor
        • Avoid smoking and limit alcohol consumption
        • Manage stress through relaxation techniques
        """
    else:
        recommendations = """
        • Continue maintaining a healthy lifestyle
        • Regular health check-ups (annual cardiac screening)
        • Maintain a balanced diet and regular exercise
        • Monitor blood pressure and cholesterol levels
        • Avoid smoking and excessive alcohol consumption
        """
    
    story.append(Paragraph(recommendations, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Footer
    footer_text = """
    <i>Note: This report is generated by an AI-based system and should not replace professional medical advice. 
    Please consult with a qualified healthcare provider for proper diagnosis and treatment.</i>
    <br/><br/>
    <b>Team:</b> Dũng, Anh, Vinh, Hằng, Huy | AIO2025 VietAI
    """
    story.append(Paragraph(footer_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


def create_feature_importance_plot(model, feature_names, model_name):
    """
    Tạo biểu đồ feature importance
    
    Args:
        model: Trained model
        feature_names (list): Tên các features
        model_name (str): Tên model
    
    Returns:
        plotly.graph_objects.Figure
    """
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None
    
    # Create dataframe
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    # Create plot
    fig = go.Figure(go.Bar(
        x=df['importance'],
        y=df['feature'],
        orientation='h',
        marker=dict(
            color=df['importance'],
            colorscale='Viridis',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title=f'Feature Importance - {model_name}',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def calculate_model_agreement(predictions):
    """
    Tính % models đồng ý với nhau
    
    Args:
        predictions (list): Danh sách binary predictions (0 hoặc 1)
    
    Returns:
        dict: Agreement statistics
    """
    if not predictions:
        return {}
    
    total_models = len(predictions)
    disease_count = sum(predictions)
    no_disease_count = total_models - disease_count
    
    agreement_pct = max(disease_count, no_disease_count) / total_models * 100
    
    return {
        'total_models': total_models,
        'disease_votes': disease_count,
        'no_disease_votes': no_disease_count,
        'agreement_percentage': agreement_pct,
        'consensus': 'Strong' if agreement_pct >= 80 else 'Moderate' if agreement_pct >= 60 else 'Weak'
    }


def export_experiments_to_excel(experiments_df, output_file="experiments_summary.xlsx"):
    """
    Export experiments to Excel với multiple sheets
    
    Args:
        experiments_df (pd.DataFrame): DataFrame chứa experiments
        output_file (str): Tên file output
    
    Returns:
        BytesIO: Excel file in memory
    """
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Sheet 1: All experiments
        experiments_df.to_excel(writer, sheet_name='All Experiments', index=False)
        
        # Sheet 2: Summary by model
        if 'model_name' in experiments_df.columns and 'accuracy' in experiments_df.columns:
            model_summary = experiments_df.groupby('model_name').agg({
                'accuracy': ['mean', 'std', 'max', 'min'],
                'f1': ['mean', 'std', 'max', 'min']
            }).round(4)
            model_summary.to_excel(writer, sheet_name='Summary by Model')
        
        # Sheet 3: Summary by dataset
        if 'dataset_type' in experiments_df.columns:
            dataset_summary = experiments_df.groupby('dataset_type').agg({
                'accuracy': ['mean', 'std', 'max', 'min'],
                'f1': ['mean', 'std', 'max', 'min']
            }).round(4)
            dataset_summary.to_excel(writer, sheet_name='Summary by Dataset')
        
        # Sheet 4: Best configurations
        best_configs = experiments_df.nlargest(10, 'accuracy')[
            ['model_name', 'dataset_type', 'accuracy', 'f1', 'auc']
        ]
        best_configs.to_excel(writer, sheet_name='Top 10 Configurations', index=False)
    
    buffer.seek(0)
    return buffer


def get_feature_descriptions():
    """
    Trả về mô tả chi tiết cho các features
    
    Returns:
        dict: Feature descriptions
    """
    return {
        'age': {
            'name': 'Age',
            'description': 'Age of the patient in years',
            'normal_range': '29-77',
            'unit': 'years'
        },
        'sex': {
            'name': 'Sex',
            'description': 'Gender of the patient',
            'values': {'0': 'Female', '1': 'Male'}
        },
        'cp': {
            'name': 'Chest Pain Type',
            'description': 'Type of chest pain experienced',
            'values': {
                '1': 'Typical Angina',
                '2': 'Atypical Angina',
                '3': 'Non-anginal Pain',
                '4': 'Asymptomatic'
            }
        },
        'trestbps': {
            'name': 'Resting Blood Pressure',
            'description': 'Resting blood pressure (on admission to hospital)',
            'normal_range': '94-200',
            'unit': 'mm Hg'
        },
        'chol': {
            'name': 'Serum Cholesterol',
            'description': 'Serum cholesterol level',
            'normal_range': '126-564',
            'unit': 'mg/dl'
        },
        'fbs': {
            'name': 'Fasting Blood Sugar',
            'description': 'Fasting blood sugar > 120 mg/dl',
            'values': {'0': 'False', '1': 'True'}
        },
        'restecg': {
            'name': 'Resting ECG',
            'description': 'Resting electrocardiographic results',
            'values': {
                '0': 'Normal',
                '1': 'ST-T wave abnormality',
                '2': 'Left ventricular hypertrophy'
            }
        },
        'thalach': {
            'name': 'Maximum Heart Rate',
            'description': 'Maximum heart rate achieved',
            'normal_range': '71-202',
            'unit': 'bpm'
        },
        'exang': {
            'name': 'Exercise Induced Angina',
            'description': 'Exercise induced angina',
            'values': {'0': 'No', '1': 'Yes'}
        },
        'oldpeak': {
            'name': 'ST Depression',
            'description': 'ST depression induced by exercise relative to rest',
            'normal_range': '0-6.2',
            'unit': 'mm'
        },
        'slope': {
            'name': 'Slope of ST Segment',
            'description': 'The slope of the peak exercise ST segment',
            'values': {
                '1': 'Upsloping',
                '2': 'Flat',
                '3': 'Downsloping'
            }
        },
        'ca': {
            'name': 'Number of Major Vessels',
            'description': 'Number of major vessels colored by fluoroscopy',
            'values': {'0': '0', '1': '1', '2': '2', '3': '3'}
        },
        'thal': {
            'name': 'Thalassemia',
            'description': 'Thalassemia blood disorder',
            'values': {
                '3': 'Normal',
                '6': 'Fixed Defect',
                '7': 'Reversible Defect'
            }
        }
    }


def get_preset_examples():
    """
    Trả về các preset examples để test
    
    Returns:
        dict: Preset patient examples
    """
    return {
        'Normal Patient': {
            'age': 45, 'sex': 1, 'cp': 4, 'trestbps': 120, 'chol': 200,
            'fbs': 0, 'restecg': 0, 'thalach': 170, 'exang': 0,
            'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3
        },
        'Low Risk': {
            'age': 50, 'sex': 0, 'cp': 3, 'trestbps': 130, 'chol': 220,
            'fbs': 0, 'restecg': 0, 'thalach': 160, 'exang': 0,
            'oldpeak': 0.5, 'slope': 1, 'ca': 0, 'thal': 3
        },
        'Medium Risk': {
            'age': 60, 'sex': 1, 'cp': 2, 'trestbps': 140, 'chol': 250,
            'fbs': 1, 'restecg': 1, 'thalach': 140, 'exang': 0,
            'oldpeak': 1.5, 'slope': 2, 'ca': 1, 'thal': 6
        },
        'High Risk': {
            'age': 65, 'sex': 1, 'cp': 1, 'trestbps': 160, 'chol': 300,
            'fbs': 1, 'restecg': 2, 'thalach': 120, 'exang': 1,
            'oldpeak': 3.0, 'slope': 3, 'ca': 3, 'thal': 7
        }
    }


# Example usage
if __name__ == "__main__":
    # Test PatientHistoryManager
    history_manager = PatientHistoryManager()
    
    # Add a sample prediction
    sample_data = {'age': 50, 'sex': 1, 'cp': 2}
    sample_predictions = [
        {'Model': 'XGBoost', 'Prediction': 'Heart Disease', 'Confidence': 0.85},
        {'Model': 'Random Forest', 'Prediction': 'Heart Disease', 'Confidence': 0.78}
    ]
    
    history_manager.add_prediction('P001', sample_data, sample_predictions, 'Heart Disease')
    
    # Get statistics
    stats = history_manager.get_statistics()
    print("Statistics:", stats)
    
    print("\n✅ app_utils.py tested successfully!")
