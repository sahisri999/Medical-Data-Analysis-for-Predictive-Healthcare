# Medical-Data-Analysis-for-Predictive-Healthcare

**Objective**
The goal of this project is to develop machine learning models to predict disease risks using patient data. We use Logistic Regression and Random Forest classifiers to analyze medical datasets and determine whether a patient is at risk of developing a disease (e.g., diabetes).

**Methodology**

Data Preprocessing:
Loaded the dataset and checked for missing values.
Standardized the features using StandardScaler.
Split the dataset into training (80%) and testing (20%) sets.

Model Training & Evaluation:
Logistic Regression and Random Forest models were trained.
The models were evaluated using:Accuracy Score
Confusion Matrix
Classification Report (Precision, Recall, F1-score)

Results Visualization:
A heatmap was used to visualize the confusion matrix.
Feature importance was analyzed for Random Forest.
Model Saving:

The trained models and scaler were saved using joblib for future use.

**Key Findings**

Random Forest performed better than Logistic Regression, as it can handle complex patterns in the data.
Glucose levels and BMI were the most important features for predicting diabetes.
The models provide a data-driven approach to assist in early disease detection.

**Conclusion**

This project demonstrates the power of machine learning in predictive healthcare by analyzing patient data to predict diabetes risk. The developed models can assist healthcare professionals in early diagnosis and intervention to improve patient outcomes.
