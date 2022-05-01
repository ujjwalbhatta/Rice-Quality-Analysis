import streamlit as st
import numpy as np
import pickle


def show_SonaMansuli_page():
    st.title("Sona Mansuli Page")

    moisture = st.number_input('Moisture', value=14.0)
    foreignMatter = st.number_input('Foreign Matter', value=0.5)
    broken = st.number_input('Broken', value=15.0)
    redAndChalky = st.number_input('Red And Chalky', value=7.0)
    damagedGrain = st.number_input('Damaged Grain', value=2.0)
    discoloredGrain = st.number_input('Discolored Grain', value=2.0)

    filename = ''

    algorithm = st.selectbox(
        "Select Algorithm Type", ("Logistics Regression", "SVM", "Naive Bayes"))

    if algorithm == "Logistics Regression":
        filename = 'model_logistic.pkl'
    elif algorithm == "SVM":
        filename = "model_svm.pkl"
    else:
        filename = "model_nb.pkl"

    def load_model():
        with open(f'./models/SonaMansuli/{filename}'.format(filename), 'rb') as file:
            data = pickle.load(file)
        return data

    model = load_model()

    error_msg = ''
    flag = True

    if(moisture > 14):
        error_msg = 'Moisture should be less than 14'
        flag = False

    if(foreignMatter > 0.5):
        error_msg = 'Foreign Matter should be less than 0.5'
        flag = False
    if(broken > 15):
        error_msg = 'Broken should be less than 15'
        flag = False
    if(redAndChalky > 7):
        error_msg = 'Red and Chalky should be less than 7'
        flag = False
    if(damagedGrain > 2):
        error_msg = 'Damaged grain should be less than 2'
        flag = False
    if(discoloredGrain > 2):
        error_msg = 'Discolored grain should be less than 2'
        flag = False
    submit = st.button("Find Grade")

    # features
    features = [moisture, foreignMatter, broken,
                redAndChalky, damagedGrain, discoloredGrain]

    if submit and flag:
        X = np.array([features])
        X = X.astype(float)

        result = model.predict(X)
        grade = ''
        if result == 0:
            grade = 'A'
        elif result == 1:
            grade = 'B'
        else:
            grade = 'C'

        st.subheader(f"The rice is of grade {grade[0]}")
    else:
        st.subheader(error_msg)
