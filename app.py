import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.header('Cars 24 Price Predictor')

df=pd.read_csv('cars_data.csv')
st.dataframe(df.head())

col1,col2=st.columns(2)

fuel_type=col1.selectbox(
    "Select the fuel type",
    ("Diesel", "Petrol", "LPG","CNG","Electric"),)

engine=col1.slider("Select the Enginer power", 500, 5000 ,step=100)

transmission=col2.selectbox(
    "Enter the Type of Transmission",
    ("Manual", "Automatic"),)

no_of_seats=col2.selectbox("Enter the number of seats",
                        (4,5,7,9,11),)

encode_dict={
    "fuel_type":{"Diesel" : 1,"Petrol" : 2,"LPG" : 3,"CNG" : 4,"Electric": 5},
    "transmission":{"Manual":1,"Automatic":2}
}

def model_predict(fuel_type,engine,transmission,no_of_seats):

    with open("car_pred","rb") as file:
        reg_model=pickle.load(file)
        input_features=[[2018.0,1,4000,fuel_type,transmission,19.70,engine,86.30,no_of_seats]]

    return reg_model.predict(input_features)

if st.button("predict Price"):

    fuel_type=encode_dict['fuel_type'][fuel_type]
    transmission=encode_dict['transmission'][transmission]

    price=model_predict(fuel_type,engine,transmission,no_of_seats)

    st.text(f'The price of the Car is {price[0].round(2)} lakh rupees')