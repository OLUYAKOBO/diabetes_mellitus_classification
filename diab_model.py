import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scal.pkl', 'rb'))
encoder = pickle.load(open ('encoder.pkl', "rb"))


st.title('*Diabetes Mellitus Prediction*')
    
def predict():
    c1,c2 = st.columns(2)
    with(c1):
        Age = st.number_input('*Please input your age*')
        Gender = st.selectbox('*Please select your gender*',['Female','Male'])
        Polyuria = st.selectbox('*Do you suffer from excessive urination?*',['No','Yes'])
        Polydipsia = st.selectbox('*Do you suffer from excessive thirst?*',['No','Yes'])
        sudden_weight_loss = st.selectbox('*Do you suffer from excessive weight loss?*',['No','Yes'])
        weakness = st.selectbox('*Do you suffer from general body weakness?*',['No','Yes'])
        Polyphagia = st.selectbox('*Do you suffer from excessive hunger?*',['No','Yes'])
        Genital_thrush = st.selectbox('*Do you suffer from genital infections?*',['No','Yes'])
        
    with(c2):
        visual_blurring = st.selectbox('*Do you experience blurred vision?*',['No','Yes'])
        Itching = st.selectbox('*Do you experience body itching?*',['No','Yes'])
        Irritability = st.selectbox('*Do you suffer from irritation?*',['No','Yes'])
        delayed_healing = st.selectbox('*Do you experience delayed healing?*',['No','Yes'])
        partial_paresis = st.selectbox('*Do you experience partial paralysis on any parts of your body?*',['No','Yes'])
        muscle_stiffness = st.selectbox('*Do you experience muscle stiffness?*',['No','Yes'])
        Alopecia = st.selectbox('*Do you experience hair loss?*',['No','Yes'])
        Obesity = st.selectbox('*Do you suffer from obesity?*',['No','Yes'])
        
        feat = np.array([Age ,Gender, Polyuria, Polydipsia, sudden_weight_loss,
        weakness, Polyphagia, Genital_thrush, visual_blurring,
        Itching, Irritability, delayed_healing, partial_paresis,
        muscle_stiffness, Alopecia, Obesity]).reshape(1,-1)
        cols = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
       'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
       'muscle stiffness', 'Alopecia', 'Obesity']
        df = pd.DataFrame(feat, columns=cols)
        
        return df
        
df = predict()
def preprocessing():
    
    df1 = df.copy()
    cat_cols = ['Gender','Polyuria','Polydipsia','sudden weight loss',
               'weakness','Polyphagia','Genital thrush','visual blurring',
               'Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity']
    encoded_data = encoder.transform(df1[cat_cols])
    dense_data = encoded_data.todense()
    df1_encoded = pd.DataFrame(dense_data, columns = encoder.get_feature_names_out())

    df1 = pd.concat([df1,df1_encoded],
                    axis = 1)
    df1.drop(cat_cols,
             axis = 1,
             inplace = True)
    
    cols = df1.columns
    df1 = scaler.transform(df1)
    df1 = pd.DataFrame(df1,columns=cols)
    return df1
df1 = preprocessing()

st.subheader('*Diagnosis*')


import time
prediction = model.predict(df1)



if st.button('*Click here to get your diagnosis*'):
    time.sleep(10)
    with st.spinner('Predicting... Please wait...'):
        if prediction == 0:
            st.success("This indiviual is not suffering from diabetes")
        else:
            st.success("This individual is suffering from diabetes")