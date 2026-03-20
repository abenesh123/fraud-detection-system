# fraud-detection-system
Financial fraud detection using ML - XGBoost, Random Forest, SHAP explainability


A machine learning system that detects fraudulent credit card transactions using multiple classification algorithms, SMOTE for class imbalance, and SHAP for explainability.

🚀 **Live Demo:** [Click Here](https://fraud-detection-system-ttwyrn7vg5khkjfdpn5bci.streamlit.app)

---

## 📌 Problem Statement

Credit card fraud is a major financial threat. Fraud transactions make up only **0.17%** of all transactions, making this a highly imbalanced classification problem. Standard accuracy metrics are misleading — the model must maximize **fraud detection (recall)** while minimizing false alarms.

---

## 🗂️ Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Features:** 30 features (V1–V28 are PCA-transformed, plus Time and Amount)
- **Class Distribution:**
  - Normal Transactions: 99.83%
  - Fraud Transactions: 0.17%

---

## 🔧 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Database | MySQL |
| Deployment | Streamlit |

---

## 🧠 ML Pipeline

```
MySQL Database
      ↓
Data Extraction (pandas + mysql-connector)
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering (hour from time)
      ↓
Train-Test Split (80/20, stratified)
      ↓
SMOTE (inside pipeline)
      ↓
Model Training
      ↓
Evaluation (Precision, Recall, F1, ROC-AUC)
      ↓
SHAP Explainability
      ↓
Streamlit Deployment
```

---

## 📊 Models Trained

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.9767 |
| **Random Forest** | **0.9850** |
| Gradient Boosting | 0.9778 |
| XGBoost (Tuned) | 0.9817 |
| LightGBM | 0.9679 |
| Isolation Forest | 0.8717 |

✅ **Best Model: Random Forest (ROC-AUC: 0.9850)**

---

## 🔑 Key Features (from SHAP Analysis)

Top fraud indicators identified:
- **V14** — strongest negative correlation with fraud
- **V12** — strong fraud indicator
- **V10** — significant fraud signal
- **V17** — important distinguishing feature
- **Amount** — transaction amount pattern

---

## 📁 Project Structure

```
fraud-detection-system/
│
├── creditcard.py          # Main ML pipeline
├── app_fraudulent.py      # Streamlit web app
├── fraud_model.pkl        # Saved best model
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/abenesh123/fraud-detection-system.git
cd fraud-detection-system
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app**
```bash
python -m streamlit run app_fraudulent.py
```

**4. Open browser at**
```
http://localhost:8501
```

---

## 🌐 Deployment

The app is deployed on **Streamlit Cloud** and accessible publicly.

🔗 **Live App:** https://fraud-detection-system-ttwyrn7vg5khkjfdpn5bci.streamlit.app

---

## 📈 EDA Highlights

- Dataset is highly imbalanced (0.17% fraud)
- No null values found
- Most transaction amounts are low with few very high outliers
- Columns V10, V12, V14, V17 are strongly correlated with fraud class
- Fraud rate varies by hour of day

---

## 🛡️ Handling Class Imbalance

Used **SMOTE (Synthetic Minority Oversampling Technique)** inside the sklearn pipeline to prevent data leakage — SMOTE is applied only on training data, never on test data.

---

## 💡 What I Learned

- Building end-to-end ML pipelines with imbalanced data
- Using SMOTE correctly inside pipelines to avoid leakage
- SHAP values for model explainability
- Hyperparameter tuning with RandomizedSearchCV
- Connecting MySQL database to Python
- Deploying ML models with Streamlit

---

## 👤 Author

**Abinesh**
- GitHub: [@abenesh123](https://github.com/abenesh123)