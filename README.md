# Student Exam Score Prediction

A machine learning project that predicts a student's expected exam score using academic, lifestyle, and support-related factors. The project includes data analysis, data cleaning, outlier handling, model training, model saving, and a professional Streamlit web app for making predictions.

## Project Overview

The goal of this project is to build a regression-based machine learning model that can estimate a student's exam score from important input features such as attendance, study hours, previous scores, tutoring sessions, resource availability, parental involvement, school type, gender, and sleep hours.

This project is useful for understanding how different student performance factors can influence exam results. It can also help students, teachers, and parents identify areas where improvement may increase academic performance.

## Key Features

- Data loading and exploration using Pandas
- Professional data visualization using Matplotlib and Seaborn
- Missing value handling
- Outlier detection and removal using the IQR method
- Categorical feature encoding
- Feature scaling using StandardScaler
- Model training using Ridge Regression
- Model evaluation using R2 Score, MAE, and RMSE
- Saved model files for reuse
- Streamlit web app for interactive exam score prediction
- Recommendation section based on user input

## Dataset

The dataset used in this project is:

```text
StudentPerformanceFactors.csv
```

It contains student-related information such as:

- Hours studied
- Attendance
- Previous scores
- Sleep hours
- Tutoring sessions
- Access to resources
- Parental involvement
- Motivation level
- Family income
- Teacher quality
- School type
- Peer influence
- Physical activity
- Learning disabilities
- Distance from home
- Gender
- Exam score

The target variable is:

```text
Exam_Score
```

## Project Files

```text
Student/
│
├── Exam_Score_Prediction2.ipynb     # Main improved machine learning notebook
├── StudentPerformanceFactors.csv    # Dataset
├── Student_Project_App.py           # Streamlit web application
├── linear_model.pkl                 # Trained Ridge Regression model
├── scaler.pkl                       # Saved StandardScaler used by the app
├── README.md                        # Project documentation
```

Optional files may also exist in the folder, such as backup notebooks or backup app files.

## Machine Learning Workflow

The project follows these steps:

1. Import required libraries
2. Load the student performance dataset
3. Perform exploratory data analysis
4. Visualize target and feature distributions
5. Detect and remove outliers
6. Handle missing values
7. Encode categorical variables
8. Select important features
9. Scale numerical features
10. Split the data into training and testing sets
11. Train the regression model
12. Evaluate model performance
13. Save the trained model and scaler
14. Build a Streamlit app for prediction

## Selected Features Used in the App

The Streamlit app uses the following features for prediction:

```text
Attendance
Hours_Studied
Previous_Scores
Tutoring_Sessions
Access_to_Resources_High
Parental_Involvement_High
Gender
School_Type
Sleep_Hours
```

These features were selected because they are easy for users to enter and are useful for predicting exam performance.

## Model Used

The final app uses a Ridge Regression model.

Ridge Regression is an improved form of Linear Regression. It helps reduce overfitting by adding regularization, which makes the model more stable and reliable on unseen data.

The model is saved as:

```text
linear_model.pkl
```

The scaler used during training is saved as:

```text
scaler.pkl
```

Both files are required by the Streamlit app.

## Model Evaluation Metrics

The model is evaluated using:

- **R2 Score**: Measures how well the model explains variation in exam scores
- **MAE**: Mean Absolute Error, showing average prediction error
- **RMSE**: Root Mean Squared Error, giving stronger penalty to larger errors

Higher R2 and lower MAE/RMSE indicate better performance.

## Streamlit App

The project includes a professional Streamlit app where users can enter student details and get a predicted exam score.

The app provides:

- Clean dashboard-style interface
- Input controls for student details
- Predicted exam score
- Score category
- Progress bar
- Important student profile metrics
- Recommendation focus based on input values
- Model input summary table

## How to Run the Project

### 1. Install Required Libraries

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib scipy
```

### 2. Run the Jupyter Notebook

Open and run:

```text
Exam_Score_Prediction.ipynb
```

This notebook performs data analysis, model training, evaluation, and saves the model files.

### 3. Run the Streamlit App

Open terminal in the project folder and run:

```bash
streamlit run Student_Project_App.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Example Prediction Flow

1. User enters attendance percentage
2. User enters study hours per day
3. User enters previous exam score
4. User selects support and school-related details
5. App scales the input using `scaler.pkl`
6. App predicts the score using `linear_model.pkl`
7. App displays the predicted score and recommendations

## Importance of Saved Files

### linear_model.pkl

This file contains the trained machine learning model. It allows the app to make predictions without retraining the model every time.

### scaler.pkl

This file contains the same scaling method used during model training. It ensures that new user input is transformed in the same way as the training data before prediction.

Without these two files, the Streamlit app will not work correctly.

## Results and Insights

From the analysis, some of the most important factors affecting exam score are:

- Attendance
- Hours studied
- Previous scores
- Tutoring sessions
- Access to resources
- Parental involvement

The project shows that academic performance is influenced by both study behavior and support systems.

## Future Scope

This project can be improved further by:

- Adding more advanced models such as Random Forest or Gradient Boosting
- Using all dataset features in the Streamlit app
- Adding user login and student record storage
- Creating downloadable prediction reports
- Deploying the app online using Streamlit Cloud
- Adding charts that compare multiple students

## Conclusion

This project successfully demonstrates how machine learning can be used to predict student exam scores based on different academic and personal factors. It includes a complete workflow from data analysis to model deployment through a Streamlit web app, making it suitable for academic submission and practical demonstration.

## Author

Created by Jeevraj Singh Sandhu as a machine learning project for student exam score prediction.
