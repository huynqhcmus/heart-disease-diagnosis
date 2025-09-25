#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run Streamlit app
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
