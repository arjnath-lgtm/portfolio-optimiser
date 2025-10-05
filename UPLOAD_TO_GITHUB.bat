@echo off
echo ====================================
echo UPLOAD TO GITHUB - STEP BY STEP
echo ====================================
echo.
echo This will guide you through uploading
echo to GitHub for Streamlit Cloud deployment
echo.
echo ====================================
echo.

echo Step 1: Create GitHub Repository
echo ====================================
echo.
echo 1. Go to: https://github.com
echo 2. Click "New repository" (green button)
echo 3. Repository name: portfolio-optimizer
echo 4. Description: Portfolio Optimizer Web App
echo 5. Make it PUBLIC (important!)
echo 6. Don't initialize with README (we have one)
echo 7. Click "Create repository"
echo.

pause

echo Step 2: Upload Files
echo ====================================
echo.
echo Upload ALL files from this folder:
echo.
echo Required files:
echo - streamlit_app.py
echo - requirements.txt
echo - portfolio_optimizer_fixed.py
echo - fundamental_analyzer_updated.py
echo - datadump.csv
echo - README.md
echo - DEPLOYMENT_GUIDE.md
echo.
echo You can drag and drop all files at once!
echo.

pause

echo Step 3: Deploy to Streamlit Cloud
echo ====================================
echo.
echo 1. Go to: https://share.streamlit.io
echo 2. Click "New app"
echo 3. Connect your GitHub account
echo 4. Select repository: portfolio-optimizer
echo 5. Main file path: streamlit_app.py
echo 6. Click "Deploy"
echo.

pause

echo Step 4: Get Your Professional URL
echo ====================================
echo.
echo After deployment, you'll get:
echo   https://portfolio-optimizer.streamlit.app
echo.
echo This is your professional shareable link!
echo.

echo ====================================
echo SUCCESS!
echo ====================================
echo.
echo Your friends can now access:
echo   https://portfolio-optimizer.streamlit.app
echo.
echo Benefits:
echo - Professional URL (not local IP)
echo - Works 24/7 (no need to keep computer on)
echo - Fast loading (cloud servers)
echo - Mobile friendly
echo - Secure HTTPS
echo.

pause
