"""
Feature Engineering Functions for Model Pipeline
Các functions này cần thiết để unpickle models từ notebook
"""

import pandas as pd
import numpy as np


def fe_basic(X):
    """
    Basic feature engineering - no advanced transformations
    Just returns original features separated into numerical and categorical
    
    Args:
        X: DataFrame with original features
        
    Returns:
        tuple: (X_transformed, numerical_features, categorical_features)
    """
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    return X.copy(), numerical_features, categorical_features


def fe_enhanced(X):
    """
    Enhanced feature engineering with polynomial features
    (May not be used in current models, but defined for compatibility)
    """
    # For now, just return basic
    return fe_basic(X)


def fe_poly_only(X):
    """
    Polynomial features only
    (May not be used in current models, but defined for compatibility)
    """
    # For now, just return basic
    return fe_basic(X)


# Make sure these functions are available when unpickling
__all__ = ['fe_basic', 'fe_enhanced', 'fe_poly_only']
