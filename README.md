# 🎓 Student Exam Score Prediction using Machine Learning

## 📌 Project Overview
This project predicts a student's exam score based on academic, personal, and environmental factors using Machine Learning.

A Linear Regression model is trained on student performance data and deployed using a Streamlit web application.

---

## 📊 Dataset Description
The dataset contains various factors that influence student performance, such as:
- Attendance
- Hours Studied
- Previous Scores
- Tutoring Sessions
- Access to Resources
- Parental Involvement
- Gender
- School Type
- Sleep Hours

Target Variable:
- **Exam_Score**

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- Joblib

---

## 🔍 Exploratory Data Analysis (EDA)
- Distribution analysis of numerical features
- Boxplots for outlier detection
- Correlation heatmap
- Feature vs target visualizations

---

## 🤖 Machine Learning Model
- Algorithm: **Linear Regression**
- Feature Scaling: **StandardScaler**
- Train-Test Split: 80% training, 20% testing
- Evaluation Metrics:
  - R² Score
  - Mean Absolute Error (MAE)

---

## 📈 Model Performance
- Selected Feature Model R²: ~0.68
- Full Feature Model R²: ~0.77

---

## 🌐 Web Application
The trained model is deployed using **Streamlit**, allowing users to input student details and predict exam scores interactively.

---

## ▶️ How to Run the Project

pip install -r requirements.txt
streamlit run app.py

