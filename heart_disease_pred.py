# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 19:07:45 2024

@author: chitv
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('C:/Users\chitv/OneDrive/Desktop/Data_Science_Internship/Disease_System/trained_model.sav','rb'))


#Creating a function for prediction

def heart_disease_prediction(input_data):
    

    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The Person does not a Heart Disease'
    else:
        return 'The Person has Heart Disease'
                          


def main():
    
    # given a title
    st.title('Heart Diseases Prediction Web App')
    
    # getting the input from the user
    
    age = st.text_input('Age: ')
    sex = st.text_input('Gender: ')
    cp = st.text_input('CP value: ')
    trestbps = st.text_input('Trestbps level: ')
    chol = st.text_input('chol level: ')
    fbs = st.text_input('fbs level: ')
    restecg = st.text_input('restecg level: ')
    thalach = st.text_input('thalach level: ')
    exang = st.text_input('exang level: ')
    oldpeak = st.text_input('Oldpeak level: ')
    slope = st.text_input('slope level: ')
    ca = st.text_input('ca level: ')
    thal = st.text_input('thal level: ')
    
    
    #code for prediction 
    diagnosis = ''
    
    # creating a button for prediction
    if st.button('Diaboties Test Result'):
        diagnosis = heart_disease_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
        
    
    st.success(diagnosis)
    
    

if __name__=='__main__':
    main()


    
    
    
    
    
    
    
    
    
    
    



