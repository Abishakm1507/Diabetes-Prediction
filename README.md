# Diabetes Prediction Machine Learning Model

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Models Evaluated](#models-evaluated)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Performance Evaluation](#performance-evaluation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)

## Introduction
This project aims to develop a machine learning model that predicts the likelihood of diabetes based on various health parameters. The model leverages historical medical data to assist healthcare professionals and individuals in making informed health decisions.

## Features
- User-friendly interface for data input
- Data preprocessing for accurate model training
- Machine learning model training using Logistic Regression
- Performance evaluation metrics
- Predictions for individual patients based on input features

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Models Evaluated
The following machine learning algorithms were evaluated for accuracy in predicting diabetes:
- Logistic Regression<br>
Accuracy: 82.46%
- KNeighbors Classifier<br>
Accuracy: 75.97%
- Support Vector Classifier (SVC)<br>
Accuracy: 79.22%
- Naive Bayes<br>
Accuracy: 75.73%
- Decision Tree Classifier<br>
Accuracy: 73.37%
- Random Forest Classifier<br>
Accuracy: 81.57%
- AdaBoost Classifier<br>
Accuracy: 77.92%
- Gradient Boosting Classifier<br>
Accuracy: 81.81%
- XGBoost Classifier<br>
Accuracy: 78.57%
- Extra Trees Classifier<br>
Accuracy: 80.51%


## Dataset
The dataset used for this project is the Pima Indians Diabetes Database, which includes medical records for female patients and features such as glucose level, blood pressure, insulin levels, BMI, age, and more. It can be found at: [Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

## Data Preprocessing
- Handling missing values
- Data normalization (if applicable)
- Feature selection to improve model accuracy
- Splitting the dataset into training and testing sets

## Model Training
The model is trained using the Logistic Regression algorithm. The steps include:
1. Loading the dataset
2. Splitting the data into training and test sets
3. Training the Logistic Regression model
4. Making predictions on the test set

## Performance Evaluation
The performance of the model is evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report
- Mean Squared Error
- R-squared Score

## Usage
To use this project:
1. Clone the repository:
   ```bash
   git clone https://github.com/Abishakm1507/Diabetes-Prediction.git
   cd Diabetes-Prediction
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Diabetes-Prediction.ipynb
4.Run the cells in the notebook to input health parameters and receive diabetes predictions.

## Future Improvements
- Enhancing the user interface
- Incorporating more advanced data preprocessing techniques
- Building a web application for user input and predictions

## Prediction Example

Using Logistic Regression to predict diabetes for a sample patient:

```python
y_predict = reg.predict([[1, 148, 72, 35, 79.799, 33.6, 0.627, 50]])
if y_predict == 1:
    print("Diabetic")
else:
    print("Non Diabetic")

