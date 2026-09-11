# 🚀 Deployment Guide: Community Detection Web Application

This guide walks you through deploying your interactive **Community Detection Web Application** live on **Render** (Recommended) or **Vercel** for 100% free hosting.

---

## 🌟 Method 1: Deploy on Render (Recommended for Python)

Render is the optimal choice for Python/Flask data science web applications.

### Step 1: Push Your Project to GitHub
1. Create a new repository on [GitHub](https://github.com/new) (e.g., `social-network-community-detection`).
2. Run in your terminal:
   ```bash
   git add .
   git commit -m "Deploy community detection web app"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/social-network-community-detection.git
   git push -u origin main
   ```

### Step 2: Create a Free Web Service on Render
1. Go to [dashboard.render.com](https://dashboard.render.com/) and sign in with GitHub.
2. Click **"New +"** $\rightarrow$ **"Web Service"**.
3. Select your repository `social-network-community-detection`.
4. Configure the settings:
   - **Name**: `community-detection-app`
   - **Environment**: `Python 3`
   - **Region**: Any (e.g., Singapore / Oregon / Frankfurt)
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: `Free`
5. Click **"Create Web Service"**.
6. Render will build and deploy your app in 2 minutes. Your live link will look like:
   `https://community-detection-app.onrender.com`

---

## ⚡ Method 2: Deploy on Vercel

### Step 1: Install Vercel CLI (or connect GitHub)
1. Go to [vercel.com](https://vercel.com) and sign in.
2. Click **"Add New..."** $\rightarrow$ **"Project"**.
3. Import your GitHub repository.
4. Keep the default settings (Vercel automatically detects `vercel.json`).
5. Click **"Deploy"**.

---

## 💻 Method 3: Test / Run Locally

To test the web app on your laptop before deploying:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the web server
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 📋 Features Included in Web App:
- ✅ **Vis.js 2D Physics Network**: Drag nodes, zoom, click, physics freeze toggle.
- ✅ **Algorithm Switcher**: Louvain, Girvan-Newman, Label Propagation (LPA), Spectral Clustering.
- ✅ **Real-Time Metrics**: Modularity ($Q$), Conductance ($\Phi$), NMI %, ARI, and execution time.
- ✅ **Multi-Algorithm Comparison Matrix**: Side-by-side benchmark table with color scoring.
- ✅ **Custom CSV Upload**: Drag-and-drop custom graph edge lists.
- ✅ **Viva Defense Guide**: Embedded popup modal for presentation preparation.
