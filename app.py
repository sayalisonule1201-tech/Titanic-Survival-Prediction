import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

with open("model/titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🚢 Titanic Survival Prediction")
st.write(
    "Machine Learning model to predict whether a Titanic passenger "
    "would have survived based on passenger information."
)

st.divider()

st.header("Enter Passenger Details")

col1, col2 = st.columns(2)

with col1:

    passenger_class = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=25.0
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=30.0
    )

with col2:

    siblings_spouses = st.number_input(
        "Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0
    )

    parents_children = st.number_input(
        "Parents / Children",
        min_value=0,
        max_value=10,
        value=0
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["S", "C", "Q"]
    )

st.divider()

if st.button("🔮 Predict Survival", use_container_width=True):

    passenger_data = pd.DataFrame({
        "Pclass": [passenger_class],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [siblings_spouses],
        "Parch": [parents_children],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    prediction = model.predict(passenger_data)[0]

    probability = model.predict_proba(passenger_data)[0]

    st.header("Prediction Result")

    if prediction == 1:

        st.success("🟢 Passenger is predicted to SURVIVE")

    else:

        st.error("🔴 Passenger is predicted NOT to survive")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Survival Probability",
            f"{probability[1] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Death Probability",
            f"{probability[0] * 100:.2f}%"
        )

st.divider()

st.header("About This Project")

st.write("""
This project uses the Titanic dataset from Kaggle to build a
machine learning classification model.

The model uses passenger class, gender, age, family information,
fare and port of embarkation to predict passenger survival.
""")

st.info(
    "Model: Random Forest Classifier | "
    "Framework: Scikit-learn | "
    "Web Application: Streamlit"
)

st.caption("Titanic Survival Prediction | Machine Learning Mini Project")
