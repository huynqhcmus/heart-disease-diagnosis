# 🚀 Deployment Guide - Streamlit Cloud

## 📦 **Deployed App**

**Live URL:** https://[your-app-name].streamlit.app

---

## 🎯 **Quick Deploy Steps**

### **1. Push to GitHub**

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Ready for Streamlit Cloud deployment"

# Create GitHub repo and push
git remote add origin https://github.com/[your-username]/heart-disease-diagnosis.git
git branch -M main
git push -u origin main
```

### **2. Deploy on Streamlit Cloud**

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click **"New app"**
4. Select:
   - **Repository:** `[your-username]/heart-disease-diagnosis`
   - **Branch:** `main`
   - **Main file:** `improved_app.py`
5. Click **"Deploy"**

⏱️ **Deployment time:** 3-5 minutes

---

## ⚠️ **Important Notes**

### **Model Files**
Models (`.pkl` files) are **NOT included in Git** due to size.

**Options:**

#### **Option 1: Use Git LFS (Recommended)**
```bash
# Install Git LFS
brew install git-lfs  # macOS
# or sudo apt-get install git-lfs  # Linux

# Track .pkl files
git lfs install
git lfs track "*.pkl"
git lfs track "*.joblib"
git add .gitattributes
git commit -m "Add Git LFS for model files"
git push
```

#### **Option 2: Use External Storage**
- Upload models to Google Drive/Dropbox
- Download in app on startup
- Add download code to `improved_app.py`

#### **Option 3: Retrain on Cloud**
- Include training notebook
- Retrain models after deployment
- Save to persistent storage

---

## 🔧 **Configuration**

### **Files for Streamlit Cloud:**
- ✅ `improved_app.py` - Main app
- ✅ `requirements.txt` - Python dependencies
- ✅ `packages.txt` - System packages (optional)
- ✅ `.streamlit/config.toml` - Streamlit config

### **Environment Variables** (if needed):
In Streamlit Cloud dashboard → Settings → Secrets:
```toml
# Add any secrets here
# Example:
# API_KEY = "your-api-key"
```

---

## 📊 **Resource Limits (Free Tier)**

- **CPU:** 1 vCPU
- **RAM:** 1 GB
- **Storage:** Limited
- **Uptime:** Sleeps after inactivity

**Tips:**
- Optimize model loading (lazy loading)
- Use caching (`@st.cache_data`, `@st.cache_resource`)
- Minimize file sizes

---

## 🐛 **Troubleshooting**

### **App won't start:**
```
1. Check logs in Streamlit Cloud dashboard
2. Verify all dependencies in requirements.txt
3. Ensure models are accessible
```

### **Out of memory:**
```
1. Reduce number of models loaded
2. Use model compression
3. Implement lazy loading
```

### **Slow loading:**
```
1. Add @st.cache_resource to model loading
2. Optimize imports
3. Use progress indicators
```

---

## 🎓 **Post-Deployment**

### **Share Your App:**
```
URL: https://[your-app-name].streamlit.app
```

### **Custom Domain** (Optional, paid):
- Streamlit Cloud Pro allows custom domains
- Example: `heart-disease.yourdomain.com`

### **Analytics:**
- Streamlit Cloud provides basic usage stats
- Add Google Analytics for detailed tracking

---

## 📝 **Maintenance**

### **Updates:**
```bash
# Make changes
git add .
git commit -m "Update feature X"
git push

# Streamlit Cloud auto-redeploys!
```

### **Rollback:**
```bash
git revert HEAD
git push
```

---

## 🎯 **Best Practices**

1. ✅ Test locally before pushing
2. ✅ Use environment variables for secrets
3. ✅ Add error handling
4. ✅ Implement caching
5. ✅ Monitor resource usage
6. ✅ Keep dependencies minimal
7. ✅ Document your app (README)

---

**🎊 Your app is now accessible worldwide!**

*Team: Dũng, Anh, Vinh, Hằng, Huy | AIO2025 VietAI*
