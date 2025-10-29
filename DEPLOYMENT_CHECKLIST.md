# 🚀 Quick Deployment Checklist

Use this checklist to deploy your dashboard step-by-step.

## ✅ Pre-Deployment

- [ ] Rename `bonus_quest_complete.py` to `app.py`
- [ ] Test locally: `streamlit run app.py`
- [ ] Verify all meetings load correctly
- [ ] Check all tabs work properly
- [ ] Review requirements.txt has all packages

## ✅ GitHub Setup

- [ ] Create GitHub account (if needed)
- [ ] Install Git or GitHub Desktop
- [ ] Create new repository: `meeting-intelligence-dashboard`
- [ ] Copy all files to repository folder:
  - [ ] app.py (your main file)
  - [ ] requirements.txt
  - [ ] README.md
  - [ ] .gitignore
  - [ ] .streamlit/config.toml
  - [ ] meetings/ folder (optional)
- [ ] Commit all files
- [ ] Push to GitHub

## ✅ Streamlit Cloud Deployment

- [ ] Go to https://streamlit.io/cloud
- [ ] Sign in with GitHub
- [ ] Click "New app"
- [ ] Select your repository
- [ ] Set main file: `app.py`
- [ ] Choose custom URL
- [ ] Click "Deploy"
- [ ] Wait 2-5 minutes
- [ ] Test live app

## ✅ Post-Deployment

- [ ] Test all features on live site
- [ ] Share URL with team
- [ ] Add demo link to README
- [ ] Star your repo ⭐
- [ ] Celebrate! 🎉

## 📝 Common Commands

### First Time Setup
```bash
# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit"

# Connect to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/meeting-intelligence-dashboard.git

# Push
git push -u origin main
```

### Future Updates
```bash
# Make changes to your files, then:
git add .
git commit -m "Description of changes"
git push
```

## 🆘 Quick Troubleshooting

**Issue**: Module not found
**Fix**: Add package to requirements.txt

**Issue**: App won't load
**Fix**: Check Streamlit Cloud logs

**Issue**: File upload fails
**Fix**: Increase maxUploadSize in config.toml

**Issue**: Models loading slowly
**Fix**: Already cached with @st.cache_resource

## 📞 Need Help?

- GitHub Issues: Create an issue in your repo
- Streamlit Forum: https://discuss.streamlit.io
- Documentation: See DEPLOYMENT_GUIDE.md

## 🎯 Your App URL

After deployment, your app will be at:
```
https://YOUR-CUSTOM-NAME.streamlit.app
```

Save this URL and share it with your team!
