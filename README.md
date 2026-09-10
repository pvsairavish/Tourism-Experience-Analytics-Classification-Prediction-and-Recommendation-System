# Tourism-Experience-Analytics-Classification-Prediction-and-Recommendation-System

# ✈️ Tourism Experience Analytics
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification%20%7C%20Regression-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
## 🚀 Live Demo

**Try the live application here:**

👉 [(https://psfupxz2vfw24yip7pljyo.streamlit.app/)](https://psfupxz2vfw24yip7pljyo.streamlit.app/)]

### Classification • Rating Prediction • Personalized Recommendation System

---

## 📌 Project Overview

**Tourism Experience Analytics** is an end-to-end Machine Learning project that helps tourism platforms and travel agencies deliver personalized experiences to users.

The system solves three main problems:

1. **Classification** → Predicts the most likely **Visit Mode** (Family, Couples, Friends, Business, Solo, etc.)
2. **Regression** → Predicts the **Rating** a user is likely to give to an attraction
3. **Recommendation** → Suggests personalized tourist attractions using Collaborative Filtering and Content-Based Filtering

Finally, a user-friendly **Streamlit web application** is built for real-time predictions and recommendations.

---

## 🎯 Business Objectives

- Provide personalized attraction recommendations
- Predict user visit mode for targeted marketing
- Estimate expected user satisfaction (rating)
- Help tourism businesses understand trends and improve offerings
- Increase customer engagement and retention

---

## 🗂️ Dataset

The project uses a multi-table tourism dataset containing:

| File                  | Description                              |
|-----------------------|------------------------------------------|
| Transaction.xlsx      | User visits, ratings, visit mode, date   |
| User.xlsx             | User demographics (Continent, Region, Country, City) |
| Item.xlsx             | Attraction details                       |
| Type.xlsx             | Attraction types                         |
| Mode.xlsx             | Visit mode categories                    |
| Continent.xlsx        | Continent names                          |
| Region.xlsx           | Region names                             |
| Country.xlsx          | Country names                            |
| City.xlsx             | City names                               |

---

## 🛠️ Tech Stack

### Modeling (Kaggle)
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM
- Scikit-Surprise (Collaborative Filtering)
- MLflow (Experiment Tracking)
- Matplotlib, Seaborn, Plotly

### Deployment (VS Code)
- Streamlit
- Joblib
- Plotly

---

## 📁 Project Structure
tourism_analytics/
│
├── notebooks/                          # Kaggle Notebook
│   └── tourism_experience_analytics.ipynb
│
├── saved_models/                       # Models & artifacts from Kaggle
│   ├── best_classification_model.pkl
│   ├── best_regression_model.pkl
│   ├── svd_collaborative.pkl
│   ├── label_encoders.pkl
│   ├── feature_columns.pkl
│   ├── attraction_features.pkl
│   ├── cosine_sim.pkl
│   └── cleaned_tourism_data.csv
│
├── app.py                              # Streamlit Application
├── requirements.txt
└── README.md
text---

## 🧠 Modeling Pipeline (Kaggle)

### 1. Data Cleaning & Merging
- Handled missing values
- Merged all tables into one master dataset
- Cleaned VisitMode and categorical columns

### 2. Exploratory Data Analysis (16 Charts)
- Rating distribution
- Visit Mode distribution
- Continent & Region analysis
- Attraction Type popularity
- Seasonal trends
- Correlation analysis
- Multivariate analysis

### 3. Feature Engineering
- Label Encoding of categorical variables
- User-level aggregated features
- Attraction-level aggregated features
- Peak season flag
- Rating difference features

### 4. Model Training

#### Classification (Predict Visit Mode)
- Random Forest
- XGBoost
- LightGBM
- Tracked using **MLflow**

#### Regression (Predict Rating)
- Random Forest
- XGBoost
- LightGBM
- Tracked using **MLflow**

#### Recommendation System
- **Collaborative Filtering** → SVD (Surprise library)
- **Content-Based Filtering** → TF-IDF + Cosine Similarity

### 5. Model Saving
All best models and preprocessing objects are saved as `.pkl` files for deployment.

---

## 🚀 Deployment (Streamlit App)

### Features of the App
- Cascading dropdowns (Continent → Region → Country → City)
- Calendar date picker (Year & Month)
- Predict Visit Mode
- Predict Expected Rating
- Personalized Attraction Recommendations
- Interactive charts and insights

---

## ⚙️ How to Run the Project

### Step 1: Modeling (Kaggle)
1. Upload the dataset to Kaggle
2. Open the notebook
3. Run all cells
4. Download the `saved_models` folder from Output

### Step 2: Deployment (VS Code / Local)

1. Create project folder and place files:
tourism_app/
├── saved_models/          ← Paste downloaded models here
├── app.py
└── requirements.txt
text2. Install dependencies:

```bash
pip install -r requirements.txt

Run the app:

Bashstreamlit run app.py

📦 requirements.txt
txtstreamlit
pandas
numpy
scikit-learn
xgboost
lightgbm
joblib
scikit-surprise
plotly

🖥️ How to Use the Streamlit App

Select Continent
Select Region (auto-filtered)
Select Country (auto-filtered)
Select City / Town (auto-filtered)
Select Preferred Attraction Type
Select Visit Date using calendar
Click Generate Predictions

You will get:

Predicted Visit Mode
Predicted Rating
Recommended Attractions
Related charts


📊 Model Performance

























TaskBest ModelKey MetricClassificationXGBoost / LightGBMAccuracy / F1-ScoreRegressionXGBoost / LightGBMR² Score / RMSERecommendationSVD + Content-BasedQualitative
Note: Actual scores depend on the final training run.

📈 Key Insights from EDA

Family and Couples are the most common visit modes
Ratings are generally high (skewed towards 4 & 5)
Clear seasonal patterns exist in tourism
Some attraction types consistently receive higher ratings
Most users have very few visits (opportunity potential)


✨ Future Improvements

Add Hybrid Recommendation System
Deploy on Streamlit Cloud / AWS
Add user login and history-based recommendations
Include maps for attractions
Real-time feedback collection


👨‍💻 Author
Punati Venkata Sai Ravish
