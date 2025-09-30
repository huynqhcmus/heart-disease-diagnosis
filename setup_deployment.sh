#!/bin/bash

echo "🚀 STREAMLIT CLOUD DEPLOYMENT SETUP"
echo "===================================="

# 1. Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git already initialized"
fi

# 2. Setup Git LFS for model files
echo ""
echo "📦 Setting up Git LFS for large model files..."
if command -v git-lfs &> /dev/null; then
    git lfs install
    git lfs track "*.pkl"
    git lfs track "*.joblib"
    echo "✅ Git LFS configured"
else
    echo "⚠️  Git LFS not installed. Model files won't be tracked."
    echo "   Install with: brew install git-lfs (macOS) or sudo apt-get install git-lfs (Linux)"
fi

# 3. Add files
echo ""
echo "📦 Adding files to Git..."
git add .
echo "✅ Files added"

# 4. Create initial commit
echo ""
echo "📦 Creating initial commit..."
git commit -m "Initial commit - Ready for Streamlit Cloud deployment

Features:
- 8 ML models for heart disease prediction
- Interactive Streamlit app with 5 tabs
- Experiment tracking and management
- Feature importance analysis
- Patient history and PDF reports

Team: Dũng, Anh, Vinh, Hằng, Huy | AIO2025 VietAI"

echo "✅ Commit created"

# 5. Instructions
echo ""
echo "===================================="
echo "✅ SETUP COMPLETE!"
echo "===================================="
echo ""
echo "📋 NEXT STEPS:"
echo ""
echo "1. Create GitHub repository:"
echo "   - Go to https://github.com/new"
echo "   - Name: heart-disease-diagnosis"
echo "   - Make it Public"
echo "   - Don't initialize with README"
echo ""
echo "2. Link to GitHub:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/heart-disease-diagnosis.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Deploy on Streamlit Cloud:"
echo "   - Go to https://share.streamlit.io/"
echo "   - Sign in with GitHub"
echo "   - Click 'New app'"
echo "   - Select your repo"
echo "   - Main file: improved_app.py"
echo "   - Click Deploy!"
echo ""
echo "===================================="
echo "🎊 Your app will be live in 3-5 minutes!"
echo "===================================="
