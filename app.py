import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
# IMPORTANT: This 'final-pipeline-salary-prediction.pkl' must be trained
#            on the features from adult 3.csv (e.g., age, workclass, education, etc.)
#            for the predictions to be accurate.
pipe = pickle.load(open('final-pipeline-salary-prediction.pkl', 'rb'))

# App title
st.title('Salary Prediction App (based on Adult Dataset)')

# Getting user input based on adult 3.csv columns
age = st.number_input('Age', min_value=17, max_value=90, value=30) # Age range typically 17-90
workclass = st.selectbox('Workclass', ['Private', 'Local-gov', '?', 'Self-emp-not-inc', 'Federal-gov', 'State-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
education = st.selectbox('Education', ['11th', 'HS-grad', 'Assoc-acdm', 'Some-college', '10th', 'Prof-school', '7th-8th', 'Bachelors', 'Masters', 'Doctorate', '5th-6th', 'Assoc-voc', '9th', '12th', '1st-4th', 'Preschool'])
marital_status = st.selectbox('Marital Status', ['Never-married', 'Married-civ-spouse', 'Widowed', 'Divorced', 'Separated', 'Married-spouse-absent', 'Married-AF-spouse'])
occupation = st.selectbox('Occupation', ['Machine-op-inspct', 'Farming-fishing', 'Protective-serv', '?', 'Other-service', 'Prof-specialty', 'Craft-repair', 'Adm-clerical', 'Exec-managerial', 'Tech-support', 'Sales', 'Priv-house-serv', 'Transport-moving', 'Handlers-cleaners', 'Armed-Forces'])
relationship = st.selectbox('Relationship', ['Own-child', 'Husband', 'Not-in-family', 'Unmarried', 'Wife', 'Other-relative'])
race = st.selectbox('Race', ['Black', 'White', 'Asian-Pac-Islander', 'Other', 'Amer-Indian-Eskimo'])
gender = st.selectbox('Gender', ['Male', 'Female'])
hours_per_week = st.number_input('Hours per Week', min_value=1, max_value=99, value=40)
native_country = st.selectbox('Native Country', ['United-States', '?', 'Peru', 'Guatemala', 'Mexico', 'Dominican-Republic', 'Ireland', 'Germany', 'Philippines', 'Thailand', 'Haiti', 'El-Salvador', 'Puerto-Rico', 'Vietnam', 'South', 'Columbia', 'Japan', 'India', 'Cambodia', 'Poland', 'Laos', 'England', 'Cuba', 'Taiwan', 'Italy', 'Canada', 'Portugal', 'China', 'Nicaragua', 'Honduras', 'Iran', 'Scotland', 'Jamaica', 'Ecuador', 'Yugoslavia', 'Hungary', 'Hong', 'Greece', 'Trinadad&Tobago', 'Outlying-US(Guam-USVI-etc)', 'France', 'Holand-Netherlands'])

# Predict button
if st.button('Predict Salary!'):
    # Create a DataFrame with the new input features
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [0], # Placeholder, as fnlwgt is usually not a user input
        'education': [education],
        'educational-num': [0], # Placeholder, as educational-num is usually derived
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [gender],
        'capital-gain': [0], # Placeholder
        'capital-loss': [0], # Placeholder
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })
    
    # Ensure columns are in the correct order as expected by the model
    # (This assumes your pipeline expects a specific column order.
    #  You might need to adjust this based on your model's training process.)
    # Example for column order if your model expects it:
    # input_data = input_data[['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
    #                          'marital-status', 'occupation', 'relationship', 'race',
    #                          'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
    #                          'native-country']]

    prediction = pipe.predict(input_data)

    st.write(f'Predicted Income: {prediction[0]}') # Income is usually categorical (<=50K, >50K)