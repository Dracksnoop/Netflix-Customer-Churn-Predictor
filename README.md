# 🎬 Netflix Customer Churn Prediction

<div align="center">

### 🎯 Predicting Customer Churn with Machine Learning to Drive Retention Strategies

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge)](https://github.com/your-username/Netflix_churn)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://netflix-customer-churn-predictor.onrender.com/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](http://kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset)

</div>

---

## 📑 Table of Contents

- [🎥 Demo](#-demo)
- [📊 Overview](#-overview)
- [💡 Motivation](#-motivation)
- [⚙️ Technical Aspect](#️-technical-aspect)
- [🔧 Installation](#-installation)
- [🚀 Run](#-run)
- [☁️ Deployment on Render](#️-deployment-on-render)
- [📁 Directory Tree](#-directory-tree)
- [✅ To Do](#-to-do)
- [🐛 Bug / Feature Request](#-bug--feature-request)
- [💻 Technologies Used](#-technologies-used)
- [👥 Team](#-team)
- [📄 License](#-license)
- [🙏 Credits](#-credits)

---

## 🎥 Demo

<div align="center">

### **🌐 [Live Demo](https://netflix-customer-churn-predictor.onrender.com/)**

![Demo Screenshot - It might take few minutes to load as I'm using free hosting]<img width="1399" height="695" alt="Image" src="https://github.com/user-attachments/assets/af1598b6-d29d-46fe-b006-4a93ae83689d" />

*Interactive Streamlit dashboard for real-time churn predictions*

</div>


---

## 📊 Overview

This project addresses the critical business problem of **customer churn** in subscription-based streaming services. The goal is to predict whether a customer will churn (cancel their subscription) based on behavioral and demographic features.

<div align="center">

| 🎯 **Target Variable** | 📈 **ML Algorithm** | 🔄 **Balancing Technique** | 🎨 **Frontend** |
|:---:|:---:|:---:|:---:|
| `churn` (binary) | XGBoost | SMOTE | Streamlit |

</div>

### 🔍 **Approach**

The solution leverages supervised machine learning with **XGBoost** as the primary classifier. The pipeline includes:

- ✨ Data preprocessing and cleaning
- 🛠️ Feature engineering (creating engagement indicators)
- ⚖️ Handling class imbalance with SMOTE
- 🎓 Model training with hyperparameter optimization
- 🌐 Interactive web deployment via Streamlit

---

## 💡 Motivation

<div align="center">

### 🎯 **Why Churn Prediction Matters**

</div>

Customer retention is **paramount** for subscription-based businesses like Netflix. Research shows:

- 💰 Acquiring new customers costs **5-25x more** than retaining existing ones
- 📈 A **5% increase** in customer retention can boost profits by **25-95%**
- 🎪 Personalized retention campaigns have **3x higher** success rates

By predicting churn early, companies can implement targeted retention campaigns, offer personalized incentives, and improve overall customer lifetime value. This project demonstrates how machine learning can transform customer data into actionable business intelligence, reducing churn rates and maximizing revenue.

---

## ⚙️ Technical Aspect

### 🏗️ **Architecture Overview**
```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐      ┌──────────────┐
│   Raw Data  │─────▶│ Preprocessing│─────▶│Feature Engineer.│─────▶│SMOTE Resample│
│  (Kaggle)   │      │  & Cleaning  │      │  + Scaling      │      │   (Balance)  │
└─────────────┘      └──────────────┘      └─────────────────┘      └───────┬──────┘
                                                                              │
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐             │
│  Streamlit  │◀─────│  Predictions │◀─────│  Trained Model  │◀────────────┘
│     UI      │      │   (Joblib)   │      │    (XGBoost)    │
└─────────────┘      └──────────────┘      └─────────────────┘
```

### 🔬 **Key Preprocessing & Feature Engineering Steps**

| Step | Description | Tool |
|------|-------------|------|
| 🧹 **Missing Values** | Imputation for numerical and categorical features | pandas |
| 🔤 **Encoding** | One-hot encoding for categorical variables | scikit-learn |
| 📏 **Scaling** | Standardization of numerical features | StandardScaler |
| 📊 **Engagement Metric** | Created `low_engagement` from `avg_watch_time_per_day` | Custom logic |
| 🎯 **Domain Features** | `support_ticket_frequency`, `device_count` indicators | Feature engineering |

### 🤖 **Modeling Pipeline**
```mermaid
graph LR
    A[📂 Dataset] --> B[🔄 SMOTE]
    B --> C[🌲 Random Forest]
    B --> D[⚡ XGBoost]
    B --> E[📈 Logistic Reg.]
    C --> F[📊 Comparison]
    D --> F
    E --> F
    F --> G[🏆 Best Model]
    G --> H[💾 Joblib Save]
```

#### 📈 **Evaluation Metrics**

<div align="center">

| Metric | Score |
|:------:|:-----:|
| 🎯 **ROC AUC** | `<ROC_AUC>` |
| ✅ **Accuracy** | `<ACCURACY>` |
| 🎪 **Precision** | `<PRECISION>` |
| 🔍 **Recall** | `<RECALL>` |
| ⚖️ **F1 Score** | `<F1_SCORE>` |

</div>

#### 🔧 **Model Development Steps**

1. **⚖️ Class Imbalance:** Applied SMOTE for balanced training
2. **🔬 Model Selection:** Compared XGBoost, Random Forest, Logistic Regression
3. **🎛️ Hyperparameter Tuning:** GridSearchCV for optimal parameters
4. **📦 Serialization:** Complete pipeline saved with Joblib

---

## 🔧 Installation

<div align="center">

### 🛠️ **Setup Instructions**

</div>

# 📥 Clone the repository
```bash
git clone https://github.com/<your-username>/Netflix_churn.git
cd Netflix_churn
```

# 🐍 Create virtual environment
```bash
python -m venv .venv
```
# ⚡ Activate virtual environment
# On Windows:
```bash
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

# 📦 Install dependencies
```bash

pip install --upgrade pip
pip install -r requirements.txt
```

### 📊 **Download Dataset**

<div align="center">

[![Kaggle Dataset](https://img.shields.io/badge/Download-Kaggle_Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](http://kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset)

</div>

**Option 1: Manual Download**

1. Visit the [dataset page](http://kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset)
2. Download `netflix_customer_churn.csv`
3. Place it in `data/netflix_customer_churn.csv`

**Option 2: Kaggle API**
```bash
# 🔑 Configure Kaggle API credentials
kaggle datasets download -d abdulwadood11220/netflix-customer-churn-dataset
unzip netflix-customer-churn-dataset.zip -d data/
```

---

## 🚀 Run

### 1️⃣ **Train the Model**

<div align="center">

![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>
```bash
jupyter notebook notebooks/01_preprocessing_and_modeling.ipynb
```

**📋 This notebook will:**

- 📂 Load and explore the dataset
- 🔧 Perform feature engineering and preprocessing
- 🤖 Train the XGBoost model with SMOTE
- 📊 Generate evaluation metrics and visualizations
- 💾 Save the model pipeline to `model/model_pipeline.pkl`

---

### 2️⃣ **Run Prediction Script**

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

</div>
```bash
python src/predict.py
```

**📝 Example Input (CSV row):**
```csv
subscription_type,monthly_revenue,avg_watch_time_per_day,total_watch_time,support_tickets_raised,device_count
Premium,15.99,120,3600,1,3
```

---

### 3️⃣ **Run Streamlit App Locally**

<div align="center">

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>
```bash
streamlit run src/app.py
```

🌐 **The app will open at:** `http://localhost:8501`

**✨ Features:**

- 📝 Input customer features via user-friendly form
- ⚡ Get instant churn predictions with probability scores
- 📊 View feature importance and model explanations
- 🎨 Interactive visualizations with Plotly

**📋 Example Input JSON:**
```json
{
  "subscription_type": "Premium",
  "monthly_revenue": 15.99,
  "avg_watch_time_per_day": 120,
  "total_watch_time": 3600,
  "support_tickets_raised": 1,
  "device_count": 3,
  "payment_method": "Credit Card"
}
```

---

## ☁️ Deployment on Render

<div align="center">

![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

### 🌐 **[View Live App](https://netflix-customer-churn-predictor.onrender.com/)**

</div>

### ⚙️ **Render Configuration**

#### 🔨 **Build Command:**
```bash
pip install -r requirements.txt
```

#### 🚀 **Start Command:**
```bash
streamlit run src/app.py --server.port=$PORT --server.address=0.0.0.0
```

### 🎛️ **Environment Settings**

| Setting | Value |
|---------|-------|
| 🐍 **Environment** | Python 3 |
| 🌍 **Region** | US West (or closest to audience) |
| 💻 **Instance Type** | Free / Starter |

### ⚠️ **Important Notes**

- 🔌 Render automatically assigns `$PORT` - the flag ensures Streamlit binds correctly
- 🌐 `--server.address=0.0.0.0` makes the app externally accessible
- 📦 Ensure `requirements.txt` includes all dependencies with pinned versions
- ⏱️ Cold start may take 30-60 seconds on free tier instances
- 🔄 Auto-deploys from GitHub on push to main branch

---

## 📁 Directory Tree
```
Netflix_churn/
│
├── 📁 .venv/                          # Virtual environment (not in Git)
│
├── 📁 data/
│   └── 📄 netflix_customer_churn.csv  # Raw dataset from Kaggle
│
├── 📁 notebooks/
│   └── 📓 01_preprocessing_and_modeling.ipynb  # EDA & Model training
│
├── 📁 src/
│   ├── 🎨 app.py                      # Streamlit web application
│   └── 🔮 predict.py                  # Batch prediction script
│
├── 📁 model/
│   └── 💾 model_pipeline.pkl          # Serialized ML pipeline
│
├── 📁 assets/
│   ├── 📊 feature_importance.html     # Interactive Plotly chart
│   ├── 📊 feature_importance.png      # Static feature importance
│   ├── 📈 roc_curve.html              # Interactive ROC curve
│   └── 📸 screenshot_demo.png         # App screenshot for README
│
├── 📄 requirements.txt                # Python dependencies
├── 📖 README.md                       # Project documentation
└── 🚫 .gitignore                      # Git ignore rules
```

---

## ✅ To Do

<div align="center">

### 🎯 **Future Enhancements**

</div>

- [ ] 🐳 **Dockerization:** Create Dockerfile for containerized deployment
- [ ] 🔄 **CI/CD Pipeline:** Set up GitHub Actions for automated testing and deployment
- [ ] 🎛️ **Hyperparameter Tuning:** Implement Bayesian optimization (Optuna) for better model performance
- [ ] 📊 **Model Monitoring:** Add MLflow or Weights & Biases for experiment tracking
- [ ] 🧪 **A/B Testing Framework:** Enable real-time model comparison
- [ ] 🔌 **API Endpoint:** Build FastAPI REST API for production integration
- [ ] 🔍 **Explainability Dashboard:** Add SHAP values for individual prediction explanations
- [ ] ⏰ **Automated Retraining:** Schedule periodic model updates with new data
- [ ] 📱 **Mobile Responsive UI:** Optimize Streamlit app for mobile devices
- [ ] 🔐 **Authentication:** Add user login for personalized dashboards

---

## 🐛 Bug / Feature Request

<div align="center">

### 💬 **We'd Love to Hear From You!**

</div>

If you encounter any bugs or have feature suggestions:

1. 🔍 Navigate to the **[Issues](https://github.com/Dracksnoop/Netflix-Customer-Churn-Predictor/issues/new)** tab
2. 🆕 Click **New Issue**
3. 📝 Provide:
   - 🐛 Bug description or feature proposal
   - 🔄 Steps to reproduce (for bugs)
   - ✅ Expected vs actual behavior
   - 📸 Screenshots if applicable

**Or submit a Pull Request!**

[![GitHub Issues](https://img.shields.io/badge/Report-Issues-red?style=for-the-badge&logo=github)](https://github.com/<your-username>/Netflix_churn/issues)
[![GitHub PRs](https://img.shields.io/badge/Submit-Pull_Request-green?style=for-the-badge&logo=github)](https://github.com/<your-username>/Netflix_churn/pulls)

---

## 💻 Technologies Used

<div align="center">

### 🛠️ **Tech Stack**

</div>

<div align="center">

#### **Languages & Core**
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

#### **Data Science & ML**
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)

#### **Visualization**
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)

#### **Frontend & Deployment**
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

#### **Tools & Utilities**
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

</div>

### 📚 **Detailed Library List**

| Category | Libraries |
|----------|-----------|
| 🐍 **Core** | Python 3.8+ |
| 📊 **Data Processing** | pandas, NumPy |
| 🤖 **Machine Learning** | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| 📈 **Visualization** | Plotly, Matplotlib, Seaborn |
| 🌐 **Web Framework** | Streamlit |
| 💾 **Model Persistence** | Joblib |
| 📓 **Development** | Jupyter Notebook, IPython |

---

## 👥 Team

<div align="center">

### 👨‍💻 **Project Creator**

**[Your Name]**

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/your-username)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:your.email@example.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://your-portfolio.com)

</div>

---

## 📄 License

<div align="center">

This project is licensed under the **MIT License**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

See the [LICENSE](LICENSE) file for details.

</div>

---

## 🙏 Credits

<div align="center">

### 📚 **Acknowledgments**

</div>

- 📊 **Dataset:** [Netflix Customer Churn Dataset](http://kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset) by Abdul Wadood on Kaggle
- 💡 **Inspiration:** Customer retention strategies and churn prediction research in subscription-based businesses
- 🛠️ **Open Source Libraries:** Special thanks to the communities behind:
  - scikit-learn
  - XGBoost
  - Streamlit
  - imbalanced-learn
  - Plotly

---

<div align="center">

### ⭐ **If you find this project useful, please consider giving it a star!** ⭐

[![Star on GitHub](https://img.shields.io/github/stars/your-username/Netflix_churn?style=social)](https://github.com/your-username/Netflix_churn)

---

**Made with ❤️ and ☕ for the Data Science Community**

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=your-username.Netflix_churn)

</div>
