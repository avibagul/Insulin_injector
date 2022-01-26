import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
from PIL import Image
import os

def train_ARIMA(df, t, p):
    model = SARIMAX(df[:t]['sugar'], 
                        order = (0, 0, 1), 
                        seasonal_order =(1, 1, 1, 12))
    trained_model = model.fit()
    
    return trained_model

def predict(df, t, p):
    
    tmodel = train_ARIMA(df, t, p)
    
    predictions = tmodel.predict(start = len(df[:t]), 
                          end = (len(df[:t])-1) + p, 
                          typ = 'levels').rename('predictions')
    
    return predictions

def evaluate(df, t_hours, p_hours, pred, col2):
    with col2:
        st.write("Evaluation Metric: RMSE")
        
        actual_values = list(df[t_hours:t_hours+p_hours]['sugar'])
        root_mean_sq = rmse(actual_values, pred)
        st.subheader(root_mean_sq)

def plot_data(df, t_hours, p, col2):
    with col2:
        fig, ax = plt.subplots()
        ax.plot(df[:t_hours]['sugar'])
        ax.plot(p)
        st.pyplot(fig)

def main():
    st.set_page_config(page_title="SARIMAX", layout = "wide")
    t1, t2, t3 = st.columns([3,2,2])
    logo = Image.open(os.path.join('logo.png'))
    t1.image(logo, width=180)
    t2.title("SARIMAX")
    
    col1, col2 = st.columns([1,1])
    
    with col1:
        st.write('Upload your data')
        uploaded_file = st.file_uploader("Choose a file")
        
        df = pd.read_csv(uploaded_file)
        st.write(df)
        
        t_hours = st.slider('Training hours', 24, 720, 48)
        p_hours = st.slider('Hours to predict?', 1, 24, 12)
        
        if st.button('Predict'):
            pred = predict(df, t_hours, p_hours)
            #data = list(df[:t_hours]['sugar']) + list(pred)
            #st.write(pred)
            temp_data = pd.DataFrame(columns = ['timestamp', 'sugar'])
            #st.write(temp_data)
            dates = list(df[t_hours:t_hours+p_hours]['timestamp'])
            #st.write(dates)
            temp_data['timestamp'] = dates
            temp_data['sugar'] = pred
            st.write(temp_data)
            
            plot_data(df, t_hours, pred, col2)
            
            evaluate(df, t_hours, p_hours, pred, col2)
        else:
            pass
    


if __name__ == '__main__':
    main()