# 📊 Customer Lifetime Value (CLV) Prediction

This project focuses on predicting Customer Lifetime Value (CLV) using machine learning techniques on an insurance dataset. The goal is to help businesses identify high-value customers and make better decisions in areas like marketing, retention, and customer segmentation.

---

## 🚀 Project Overview

Customer Lifetime Value is an important business metric that estimates how much revenue a customer can generate over time. In this project, I built an end-to-end ML pipeline to predict CLV based on customer demographics, policy details, and claim history.

---

## 🧠 Key Steps

### 1. Exploratory Data Analysis (EDA)
- Analyzed data distributions and relationships  
- Identified skewness in the target variable  
- Used visualizations like histograms and heatmaps  

### 2. Data Preprocessing
- Handled missing values  
- Encoded categorical variables  
- Scaled numerical features using pipelines  

### 3. Feature Engineering
- Created new features such as:
  - Claim-to-Premium Ratio  
  - Income-to-Premium Ratio  

### 4. Handling Skewness
- Applied **log transformation** on the target variable (CLV)  
- Improved model performance and stability  

### 5. Model Building
- Used **Random Forest Regressor**  
- Built pipeline using `ColumnTransformer`  
- Trained and evaluated model using R² score  

---

## 📈 Results

- Improved performance after log transformation  
- Model was able to capture non-linear relationships effectively  
- Achieved a good R² score (can mention exact value if needed)

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Streamlit  
- Hugging Face  

---

## 🌐 Deployment

- Built an interactive UI using **Streamlit**  
- Also deployed using **Hugging Face Spaces**  
- Users can input customer details and get real-time CLV predictions  

---

## 💡 What I Learned

- Importance of understanding data before modeling  
- Handling skewed data using transformations  
- Feature engineering for better performance  
- Building end-to-end ML pipelines  
- Deploying models for real-world use  

---

## 🔮 Future Improvements

- Try advanced models like XGBoost  
- Hyperparameter tuning  
- Add more business-driven features  
- Improve UI/UX of the application  

---

## 🤝 Connect

If you have suggestions or feedback, feel free to reach out!
