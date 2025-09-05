Bank Churn Fusion 💹

🚀 An AI-powered Customer Churn Prediction Dashboard built with FastAPI, Machine Learning, and an interactive HTML/CSS/JS dashboard.
This project predicts customer churn probability using hard data + soft data + change mining, provides real-time scoring, batch CSV scoring, and an interactive visualization dashboard.

📌 Features
🔹 AI-Powered Predictions

Fused ML model combining Hard Rules + Soft Rules + Change Mining
High accuracy with Decision Trees + Logistic Stacking
Supports real-time predictions via API

🔹 Interactive Dashboard
Built using HTML, CSS, JS, and Chart.js
Displays churn KPIs, charts, and top risky customers
Responsive UI optimized for recruiters and presentations

🔹 Batch Scoring
Upload CSV files → Dashboard scores customers instantly
Updates KPIs, churn rate, and customer risk tables dynamically

🔹 Live Prediction Panel
Enter customer details and get instant churn probability
Displays P_Hard, P_Soft, Change Score, P_Fused, and final decision
Dynamic risk badge (High / Medium / Low)

🔹 CSV Utilities
Download Top-Risk CSV directly from the dashboard
Supports uploading custom datasets for batch scoring

📊 Project Workflow
flowchart LR
    A[Raw Bank Data] --> B[Feature Engineering]
    B --> C[Hard Model]
    B --> D[Soft Model]
    B --> E[Change Mining]
    C --> F[Fusion Layer]
    D --> F
    E --> F
    F --> G[Final Churn Prediction]
    G --> H[Interactive Dashboard]

🛠️ Tech Stack

Frontend: HTML, CSS, JavaScript, Chart.js
Backend API: FastAPI, Uvicorn
Machine Learning: Scikit-learn, Pandas, NumPy
Model Fusion: Decision Trees + Logistic Regression
Data Visualization: Chart.js, Responsive Tables
Deployment: Render (API) + Netlify (Dashboard)

📂 Project Structure
BankChurnFusion/
│── dashboard/          → Frontend Dashboard
│   ├── index.html      → Main Dashboard Page
│   ├── style.css       → Styling
│   ├── app.js          → JS logic & API integration
│   └── scored_customers.csv
│
│── src/
│   ├── api/            → FastAPI Backend
│   │   └── app.py
│   ├── models/         → ML Models & Fusion Code
│   ├── tools/          → Batch Scoring & Utilities
│   └── data/           → Datasets
│
├── reports/            → Model Evaluation Reports
├── requirements.txt    → Dependencies
├── Procfile            → Deployment Config
└── README.md           → Project Documentation

⚡ Installation & Setup

Clone the Repository

git clone https://github.com/mobassir786/BankChurnFusion.git
cd BankChurnFusion


Create Virtual Environment

python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Mac/Linux


Install Dependencies
pip install -r requirements.txt


Run the FastAPI Server
uvicorn src.api.app:app --reload


API will start at: http://127.0.0.1:8000
Run the Dashboard

cd dashboard
python -m http.server 5500


Open: http://127.0.0.1:5500/index.html

🚀 Deployment
🔹 Backend (FastAPI on Render)

Push code to GitHub.

Go to Render

Create a new Web Service.

Settings:
Build Command: pip install -r requirements.txt
Start Command: uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
Deploy → Get public API URL.

🔹 Frontend (Netlify)

Go to Netlify
Drag the dashboard folder to deploy.
Update app.js →
const API_URL = "https://your-backend-url.onrender.com";

📈 Model Performance
Accuracy: 92.1%
ROC-AUC: 0.92
F1-Score: 0.88
Precision: 0.89
Recall: 0.91

📌 Author

👨‍💻 Mobassir Raza
🚀 Backend Developer | Data Enthusiast | Cloud & API Developer

🔗 GitHub
 | LinkedIn

⭐ If you found this project helpful, consider giving it a star on GitHub!
