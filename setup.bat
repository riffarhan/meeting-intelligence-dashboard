@echo off
REM Meeting Intelligence Dashboard - Setup Script for Windows

echo ========================================
echo Meeting Intelligence Dashboard
echo Quick Setup Script for Windows
echo ========================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed!
    echo Please install Git from: https://git-scm.com/downloads
    pause
    exit /b 1
)

echo [OK] Git is installed
echo.

REM Get user input
set /p github_username="Enter your GitHub username: "
set /p repo_name="Enter repository name (press Enter for default): "
if "%repo_name%"=="" set repo_name=meeting-intelligence-dashboard

echo.
echo Configuration:
echo   GitHub Username: %github_username%
echo   Repository Name: %repo_name%
echo.

set /p confirm="Is this correct? (y/n): "
if /i not "%confirm%"=="y" (
    echo Cancelled. Please run the script again.
    pause
    exit /b 0
)

echo.
echo [STEP 1] Setting up Git repository...

REM Initialize git if not already initialized
if not exist .git (
    git init
    echo [OK] Git initialized
) else (
    echo [OK] Git already initialized
)

REM Check if app.py exists
if not exist app.py (
    if exist bonus_quest_complete.py (
        echo [STEP 2] Copying bonus_quest_complete.py to app.py...
        copy bonus_quest_complete.py app.py
        echo [OK] app.py created
    ) else (
        echo [WARNING] app.py not found!
        echo Please make sure your main file is named app.py
    )
)

REM Check for requirements.txt
if not exist requirements.txt (
    echo [STEP 3] Creating requirements.txt...
    (
        echo streamlit^>=1.28.0
        echo pandas^>=2.0.0
        echo numpy^>=1.24.0
        echo plotly^>=5.17.0
        echo python-docx^>=1.1.0
        echo torch^>=2.0.0
        echo transformers^>=4.30.0
    ) > requirements.txt
    echo [OK] requirements.txt created
)

REM Add all files
echo.
echo [STEP 4] Adding files to git...
git add .
echo [OK] Files added

REM Commit
echo.
set /p commit_msg="Enter commit message (press Enter for default): "
if "%commit_msg%"=="" set commit_msg=Initial commit
git commit -m "%commit_msg%"
echo [OK] Files committed

REM Set up remote
echo.
echo [STEP 5] Setting up GitHub remote...
git remote remove origin 2>nul
git remote add origin https://github.com/%github_username%/%repo_name%.git
echo [OK] Remote configured

REM Switch to main branch
git branch -M main
echo [OK] Switched to main branch

echo.
echo ========================================
echo Ready to push to GitHub!
echo ========================================
echo.
echo IMPORTANT: Before pushing, make sure you have:
echo   1. Created the repository on GitHub
echo   2. Go to: https://github.com/new
echo   3. Name: %repo_name%
echo   4. Don't initialize with README
echo.

set /p repo_created="Have you created the repository on GitHub? (y/n): "
if /i not "%repo_created%"=="y" (
    echo.
    echo Please create the repository first, then run:
    echo   git push -u origin main
    echo.
    pause
    exit /b 0
)

REM Push to GitHub
echo.
echo [STEP 6] Pushing to GitHub...
git push -u origin main

if errorlevel 1 (
    echo [ERROR] Push failed!
    echo Common reasons:
    echo   - Repository doesn't exist on GitHub
    echo   - Wrong repository name
    echo   - Authentication required
    echo.
    echo Try running manually:
    echo   git push -u origin main
) else (
    echo.
    echo ========================================
    echo SUCCESS! Your code is now on GitHub!
    echo ========================================
    echo.
    echo Repository: https://github.com/%github_username%/%repo_name%
    echo.
    echo Next Steps:
    echo   1. Go to: https://streamlit.io/cloud
    echo   2. Sign in with GitHub
    echo   3. Click 'New app'
    echo   4. Select repository: %repo_name%
    echo   5. Main file: app.py
    echo   6. Click 'Deploy'
    echo.
    echo Your app will be live in 2-5 minutes!
)

echo.
pause
