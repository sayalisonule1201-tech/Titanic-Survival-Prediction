# 🚢 Titanic Survival Prediction

A machine learning web application that predicts whether a passenger would have survived the Titanic disaster based on passenger information such as gender, age, passenger class, fare and family information.

## 📌 Project Overview

This project uses the Kaggle Titanic dataset to build a classification model for predicting passenger survival.

The machine learning model is implemented using Python and Scikit-learn, while Streamlit is used to create an interactive web application.

## 🎯 Objective

The objective of this project is to:

* Analyze Titanic passenger data
* Handle missing values
* Preprocess numerical and categorical features
* Train a machine learning classification model
* Predict passenger survival
* Display predictions through an interactive Streamlit application

## 📊 Dataset

The project uses the Titanic dataset from Kaggle.

The main features used by the model are:

* Passenger Class
* Gender
* Age
* Siblings / Spouses
* Parents / Children
* Fare
* Port of Embarkation

The target variable is:

* `Survived`

  * `0` = Did not survive
  * `1` = Survived

## 🤖 Machine Learning Model

The project uses:

**Random Forest Classifier**

The preprocessing pipeline includes:

* Missing value handling
* Median imputation for numerical features
* Most-frequent imputation for categorical features
* One-hot encoding for categorical variables

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* Git
* GitHub

## 📁 Project Structure

```text
Titanic-Survival-Prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── model/
│   └── titanic_model.pkl
│
└── Images/
    └── dashboard.png
```

## 🚀 How to Run

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 🌐 Streamlit Application

The application allows users to enter passenger information and receive a predicted survival result along with survival probability.

## 📌 Future Improvements

* Add exploratory data analysis
* Add survival visualizations
* Compare multiple classification algorithms
* Add model performance metrics
* Add feature importance visualization
* Improve Streamlit dashboard design

## 👩‍💻 Author

**Sayali Sachin Sonule**

Computer Engineering Student

GitHub: https://github.com/sayalisonule1201-tech
