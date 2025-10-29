# 🚀 Deployment Guide: Meeting Intelligence Dashboard

## Overview
This guide will walk you through deploying your Streamlit dashboard to GitHub and Streamlit Cloud.

## 📋 Prerequisites

1. **GitHub Account**: Sign up at https://github.com
2. **Git Installed**: Download from https://git-scm.com/downloads
3. **Python 3.8+**: Make sure Python is installed on your system

---

## 🗂️ Step 1: Organize Your Project Structure

Your project should look like this:

```
meeting-intelligence-dashboard/
│
├── app.py                          # Your main Streamlit app (rename bonus_quest_complete.py)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Files to exclude from Git
├── .streamlit/
│   └── config.toml                 # Streamlit configuration (optional)
│
└── meetings/                       # Your meeting transcripts (optional - can be uploaded later)
    ├── Alex_Rodriguez_15Oct24.docx
    ├── Javier_Morales_22Oct24.docx
    └── ...
```

---

## 📦 Step 2: Create Required Files

### 2.1 requirements.txt
This file lists all Python packages needed:

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
python-docx>=1.1.0
torch>=2.0.0
transformers>=4.30.0
```

### 2.2 README.md
Create a compelling README:

```markdown
# 🎯 Meeting Intelligence Dashboard

AI-powered analytics for 1:1 and team meetings using the CHAT + ACT frameworks.

## Features

- **1:1 Meeting Analysis**: CHAT framework, ACT predictions, psychological safety
- **Team Meeting Analysis**: Participation balance, inclusion scores, collaboration patterns
- **Advanced NLP**: Sentiment analysis, emotion detection, linguistic diversity
- **Coaching Insights**: AI-powered recommendations for leadership development

## Quick Start

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
4. Upload your meeting transcripts

## Meeting Format

**1:1 Meetings**: `FirstName_LastName_DDMmmYY.docx` with "Name:" speaker format
**Team Meetings**: Any `.docx` with `[Speaker N]` format

## Live Demo

[View Dashboard](https://your-app-name.streamlit.app)

## License

MIT License
```

### 2.3 .gitignore
Prevent uploading unnecessary files:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Streamlit
.streamlit/secrets.toml

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Meeting files (optional - remove if you want to include them)
meetings/*.docx

# Data
*.csv
*.xlsx
```

### 2.4 .streamlit/config.toml (Optional)
Customize your Streamlit app appearance:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
```

---

## 🐙 Step 3: Upload to GitHub

### Option A: Using GitHub Desktop (Easiest)

1. **Download GitHub Desktop**: https://desktop.github.com/
2. **Create New Repository**:
   - Click "File" → "New Repository"
   - Name: `meeting-intelligence-dashboard`
   - Description: "AI-powered meeting analytics dashboard"
   - Choose local path where your project is
   - Click "Create Repository"
3. **Add Files**:
   - Copy all your files to the repository folder
   - GitHub Desktop will show all new files
4. **Commit Changes**:
   - Write commit message: "Initial commit: Complete dashboard"
   - Click "Commit to main"
5. **Publish to GitHub**:
   - Click "Publish repository"
   - Choose public or private
   - Click "Publish Repository"

### Option B: Using Command Line

1. **Open Terminal/Command Prompt** in your project folder

2. **Initialize Git**:
```bash
git init
```

3. **Add files**:
```bash
git add .
```

4. **Commit**:
```bash
git commit -m "Initial commit: Complete meeting intelligence dashboard"
```

5. **Create GitHub Repository**:
   - Go to https://github.com/new
   - Name: `meeting-intelligence-dashboard`
   - Don't initialize with README (you already have one)
   - Click "Create repository"

6. **Connect and Push**:
```bash
git remote add origin https://github.com/YOUR_USERNAME/meeting-intelligence-dashboard.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## ☁️ Step 4: Deploy to Streamlit Cloud

### 4.1 Sign Up for Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Click "Sign up" and authenticate with GitHub

### 4.2 Deploy Your App

1. **Click "New app"**

2. **Configure deployment**:
   - **Repository**: Select `YOUR_USERNAME/meeting-intelligence-dashboard`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose your custom URL (e.g., `meeting-dashboard`)

3. **Advanced Settings** (Optional):
   - **Python version**: 3.11
   - **Secrets**: Add any API keys if needed

4. **Click "Deploy"**

5. **Wait 2-5 minutes** for deployment to complete

6. **Your app is live!** 🎉
   - URL will be: `https://YOUR-APP-NAME.streamlit.app`

---

## 📤 Step 5: Update Your App

Whenever you make changes:

### Using GitHub Desktop:
1. Make changes to your files
2. Commit changes with descriptive message
3. Click "Push origin"
4. Streamlit Cloud auto-deploys in ~2 minutes

### Using Command Line:
```bash
git add .
git commit -m "Add new feature: XYZ"
git push
```

---

## 🔧 Troubleshooting

### Issue: "Module not found" error

**Solution**: Make sure all packages are in `requirements.txt`

### Issue: App crashes on Streamlit Cloud

**Solutions**:
1. Check logs in Streamlit Cloud dashboard
2. Verify `requirements.txt` has correct versions
3. Test locally first: `streamlit run app.py`

### Issue: Large file upload errors

**Solution**: Add to `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 500
```

### Issue: Slow model loading

**Solution**: Use caching (already implemented with `@st.cache_resource`)

---

## 📊 Step 6: Share Your Dashboard

### Option 1: Public Link
Share your Streamlit Cloud URL directly:
```
https://your-app-name.streamlit.app
```

### Option 2: Embed in Website
Add iframe to your website:
```html
<iframe src="https://your-app-name.streamlit.app" 
        width="100%" height="800px">
</iframe>
```

### Option 3: Add to Portfolio
Link from your GitHub README or portfolio site

---

## 🎨 Customization Tips

### Change App Name
In `app.py`, modify:
```python
st.set_page_config(
    page_title="Your Custom Title",
    page_icon="🎯",
)
```

### Add Authentication
Install `streamlit-authenticator`:
```python
import streamlit_authenticator as stauth
```

### Connect to Cloud Storage
For large datasets, use:
- **AWS S3**: `boto3`
- **Google Drive**: `gdown`
- **Dropbox**: `dropbox`

---

## 📈 Monitoring & Analytics

### Streamlit Cloud Metrics
- View app usage in Streamlit Cloud dashboard
- Monitor errors and logs
- See deployment history

### Add Google Analytics
Add to `app.py`:
```python
import streamlit.components.v1 as components

# Add at the top of your app
components.html("""
<script async src="https://www.googletagmanager.com/gtag/js?id=YOUR-GA-ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'YOUR-GA-ID');
</script>
""")
```

---

## 🔐 Security Best Practices

1. **Never commit sensitive data**:
   - Add to `.gitignore`
   - Use Streamlit secrets for API keys

2. **Use Streamlit Secrets**:
   - In Streamlit Cloud: Settings → Secrets
   - Access in code:
   ```python
   import streamlit as st
   api_key = st.secrets["api_key"]
   ```

3. **Private repositories**:
   - Keep repo private if containing sensitive data
   - Streamlit Cloud works with private repos

---

## 🚀 Next Steps

1. ✅ Deploy your dashboard
2. 📱 Test on mobile devices
3. 📊 Gather user feedback
4. 🔄 Iterate and improve
5. 📢 Share with your team!

---

## 📞 Support

- **Streamlit Docs**: https://docs.streamlit.io
- **GitHub Issues**: Create issues in your repo
- **Streamlit Forum**: https://discuss.streamlit.io

---

## 🎉 Congratulations!

You've successfully deployed your Meeting Intelligence Dashboard! 

**Your dashboard is now:**
- ✅ Live on the internet
- ✅ Accessible via custom URL
- ✅ Auto-updating on every push
- ✅ Ready to analyze meetings!

Happy analyzing! 🎯
