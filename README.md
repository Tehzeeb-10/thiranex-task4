# Real-World Health Data Project: Stress-Lysis Health Monitoring & Stress Level Classification

## 📌 Project Overview

This project was completed as a **Real-World Data Project** focused on the **healthcare domain**.

The project uses the **Stress-Lysis dataset** containing physiological and physical activity measurements to classify stress levels into **Low, Normal, and High Stress** categories.

The project follows an end-to-end data science workflow:

**Data Loading → Data Analysis → Visualization → Machine Learning → Model Evaluation → New Input Prediction**

The goal is to demonstrate how data science and machine learning can be applied to a real-world health monitoring problem.

## 🎯 Objectives

* Analyze physiological and activity-related data.
* Explore patterns and relationships between the features.
* Visualize stress-level distributions.
* Build a machine learning classification model.
* Evaluate the model using accuracy, precision, recall, and F1-score.
* Identify the importance of each input feature.
* Predict the stress level for new user-provided sensor values.

## 📊 Dataset

The dataset contains **2,001 records and 4 columns**.

### Features

| Feature      | Description                          |
| ------------ | ------------------------------------ |
| Humidity     | Sweat/humidity measurement in mg/min |
| Temperature  | Body temperature in °F               |
| Step count   | Physical activity in steps/min       |
| Stress Level | Target stress category               |

### Stress Classes

* **0 → Low Stress**
* **1 → Normal Stress**
* **2 → High Stress**

### Dataset Statistics

* Records: **2,001**
* Features: **4**
* Missing values: **0**

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Random Forest Classifier

## 🔍 Exploratory Data Analysis

The following analyses were performed:

* Dataset shape and structure analysis
* Missing-value analysis
* Statistical summary
* Stress-level distribution
* Correlation analysis
* Humidity vs Temperature analysis
* Step-count distribution across stress levels
* Feature relationship analysis

## 📈 Visualizations

The project includes visualizations for:

1. Stress Level Distribution
2. Correlation Matrix
3. Humidity vs Temperature by Stress Level
4. Step Count Distribution Across Stress Levels

The visualization output is saved as:

`stress_lysis_analysis.png`

## 🤖 Machine Learning

A **Random Forest Classifier** was used to classify the stress level.

### Train-Test Split

* Training data: **70%**
* Testing data: **30%**
* Stratified splitting was used to maintain the class distribution.

### Model Accuracy

**99.83%**

### Classification Performance

| Stress Level  | Precision | Recall | F1-Score |
| ------------- | --------: | -----: | -------: |
| Low Stress    |      0.99 |   1.00 |     1.00 |
| Normal Stress |      1.00 |   1.00 |     1.00 |
| High Stress   |      1.00 |   1.00 |     1.00 |

## ⭐ Feature Importance

The Random Forest model identified the following feature importance values:

1. **Temperature — 41.23%**
2. **Humidity — 40.15%**
3. **Step count — 18.63%**

Temperature and humidity contributed more strongly to the model's classification than step count in this dataset.

## 🧪 New Input Prediction

The project also allows the user to enter new sensor values through the Python console.

The user provides:

* Humidity
* Body Temperature
* Step Count

The trained Random Forest model then predicts:

* **Low Stress**
* **Normal Stress**
* **High Stress**

Example workflow:

```text
Enter Humidity (mg/min): 25
Enter Body Temperature (°F): 95
Enter Step Count (steps/min): 160

Predicted Stress Level: High Stress
Stress Class: 2
```

This makes the project interactive and demonstrates how a trained classification model can be used with new input data.

## 🌍 Real-World Application

This type of classification system can be considered for **health-monitoring and IoMT applications**, where physiological and activity-related measurements are collected from wearable or sensor-based devices.

A possible future application is a wearable system that collects sensor readings and provides an estimated stress category.

This project is a demonstration of machine-learning classification and should not be considered a medical diagnostic system.

## 💡 Key Findings

* The dataset contains **2,001 records with no missing values**.
* The dataset contains three stress categories: Low, Normal, and High.
* The Random Forest classifier achieved **99.83% test accuracy**.
* Temperature had the highest feature importance at **41.23%**.
* Humidity had the second-highest feature importance at **40.15%**.
* Step count contributed **18.63%** to the model.
* The trained model can classify new user-provided sensor values.

## 📁 Project Files

```text
Stress-Lysis-Health-Monitoring/
│
├── stress_analysis.py
├── Stress-Lysis.csv
├── stress_lysis_analysis.png
└── README.md
```

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

* Python-based data analysis
* Data preprocessing
* Exploratory Data Analysis
* Data visualization
* Classification using Random Forest
* Model evaluation
* Feature importance analysis
* Interactive prediction using new inputs
* Applying data science to a real-world healthcare problem

## 📌 Conclusion

This project demonstrates an end-to-end application of data science to a real-world health monitoring problem. Physiological and activity-related features were analyzed and used to train a Random Forest classifier for stress-level classification.

The model achieved **99.83% test accuracy** on the available dataset, and the completed system can also accept new sensor values and generate a predicted stress category.
