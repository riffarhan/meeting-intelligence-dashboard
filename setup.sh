#!/bin/bash

# 🚀 Meeting Intelligence Dashboard - Quick Setup Script
# This script helps you set up and deploy your dashboard quickly

echo "🎯 Meeting Intelligence Dashboard - Setup Script"
echo "================================================"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first:"
    echo "   https://git-scm.com/downloads"
    exit 1
fi

echo "✅ Git is installed"
echo ""

# Get user input
read -p "Enter your GitHub username: " github_username
read -p "Enter repository name (default: meeting-intelligence-dashboard): " repo_name
repo_name=${repo_name:-meeting-intelligence-dashboard}

echo ""
echo "📝 Configuration:"
echo "   GitHub Username: $github_username"
echo "   Repository Name: $repo_name"
echo ""

read -p "Is this correct? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Cancelled. Please run the script again."
    exit 0
fi

echo ""
echo "🔧 Setting up Git repository..."

# Initialize git if not already initialized
if [ ! -d .git ]; then
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git already initialized"
fi

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo "Creating .gitignore..."
    cat > .gitignore << EOL
__pycache__/
*.py[cod]
.Python
venv/
.streamlit/secrets.toml
.DS_Store
EOL
    echo "✅ .gitignore created"
fi

# Check if app.py exists, if not, look for bonus_quest_complete.py
if [ ! -f app.py ]; then
    if [ -f bonus_quest_complete.py ]; then
        echo "📄 Renaming bonus_quest_complete.py to app.py..."
        cp bonus_quest_complete.py app.py
        echo "✅ app.py created"
    else
        echo "⚠️  Warning: app.py not found. Make sure your main file is named app.py"
    fi
fi

# Check for requirements.txt
if [ ! -f requirements.txt ]; then
    echo "⚠️  Warning: requirements.txt not found"
    echo "Creating basic requirements.txt..."
    cat > requirements.txt << EOL
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
python-docx>=1.1.0
torch>=2.0.0
transformers>=4.30.0
EOL
    echo "✅ requirements.txt created"
fi

# Add all files
echo ""
echo "📦 Adding files to git..."
git add .

# Commit
echo ""
read -p "Enter commit message (default: Initial commit): " commit_msg
commit_msg=${commit_msg:-Initial commit}
git commit -m "$commit_msg"
echo "✅ Files committed"

# Set up remote
echo ""
echo "🔗 Setting up GitHub remote..."
git remote remove origin 2>/dev/null
git remote add origin "https://github.com/$github_username/$repo_name.git"
echo "✅ Remote configured"

# Check if main branch exists, if not create it
current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
    git branch -M main
    echo "✅ Switched to main branch"
fi

echo ""
echo "🚀 Ready to push to GitHub!"
echo ""
echo "⚠️  IMPORTANT: Before pushing, make sure you have:"
echo "   1. Created the repository on GitHub: https://github.com/new"
echo "   2. Named it: $repo_name"
echo ""
read -p "Have you created the repository on GitHub? (y/n): " repo_created

if [ "$repo_created" != "y" ]; then
    echo ""
    echo "Please create the repository first:"
    echo "1. Go to: https://github.com/new"
    echo "2. Name: $repo_name"
    echo "3. Don't initialize with README"
    echo "4. Create repository"
    echo ""
    echo "Then run: git push -u origin main"
    exit 0
fi

# Push to GitHub
echo ""
echo "📤 Pushing to GitHub..."
if git push -u origin main; then
    echo "✅ Successfully pushed to GitHub!"
    echo ""
    echo "🎉 Success! Your code is now on GitHub!"
    echo "   Repository: https://github.com/$github_username/$repo_name"
    echo ""
    echo "📍 Next Steps:"
    echo "   1. Go to: https://streamlit.io/cloud"
    echo "   2. Sign in with GitHub"
    echo "   3. Click 'New app'"
    echo "   4. Select your repository: $repo_name"
    echo "   5. Main file: app.py"
    echo "   6. Click 'Deploy'"
    echo ""
    echo "⏱️  Your app will be live in 2-5 minutes!"
else
    echo "❌ Push failed. Common reasons:"
    echo "   - Repository doesn't exist on GitHub"
    echo "   - Wrong repository name"
    echo "   - Authentication required"
    echo ""
    echo "Try running manually:"
    echo "   git push -u origin main"
fi
