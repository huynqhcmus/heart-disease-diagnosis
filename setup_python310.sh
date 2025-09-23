#!/bin/bash

echo "🐍 Setting up Python 3.10 environment"
echo "======================================"

# Check if Python 3.10 exists
if [ -f "/opt/homebrew/bin/python3.10" ]; then
    PYTHON_PATH="/opt/homebrew/bin/python3.10"
elif [ -f "/usr/local/bin/python3.10" ]; then
    PYTHON_PATH="/usr/local/bin/python3.10"
else
    echo "❌ Python 3.10 not found!"
    echo "Please run: brew install python@3.10"
    exit 1
fi

echo "✅ Found Python 3.10 at: $PYTHON_PATH"

# Remove old venv
rm -rf venv

# Create new venv
echo "Creating virtual environment..."
$PYTHON_PATH -m venv venv

# Activate
source venv/bin/activate

# Check version
echo "Python version: $(python --version)"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install packages
echo "Installing packages..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo "To activate: source venv/bin/activate"
