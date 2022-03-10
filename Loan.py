# ====================================================================
# Libraries loading
# ====================================================================
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import requests
import pickle
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
pd.set_option('display.max_colwidth', -1)

data_test = pd.read_pickle("data/data_test_ltd.pkl")
shap_values = joblib.load('data/shap_values_ltd.pkl')
file_dash_df = 'data/dash_df.pkl'
with open(file_dash_df, 'rb') as dash_df:
            dash_df = pickle.load(dash_df) 
file_exp_value = 'data/exp_value.pkl'
with open(file_exp_value, 'rb') as exp_value:
            exp_value = pickle.load(exp_value) 

# ====================================================================
# HEADER - TITRE
# ====================================================================
html_header="""
    <head>
        <title>Interactive Dashboard - Loan approval tool</title>
        <meta charset="utf-8">
        <meta name="keywords" content="Dashboard, loan, credit score, defaulters">
        <meta name="description" content="Credit scoring - dashboard">
        <meta name="author" content="Stephane Lanchec">
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>             
    <h1 style="font-size:250%; color:Black; margin-top: -100px; font-family:Arial"> Loan Approval <br>
        <h2 style="font-size:70%; "color:Gray; font-family:Georgia"> INTERACTIVE DASHBOARD</h2>
        <hr style= "  display: block;
          margin-top: 0px;
          margin-bottom: 0;
          margin-left: 0;
          margin-right: auto;
          border-style: inset;
          border-width: 2.5px;"/> 
     </h1>
"""
st.set_page_config(page_title="Prêt à dépenser - Dashboard", page_icon="", layout="wide")
st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)
st.markdown(html_header, unsafe_allow_html=True)

# Display the logo in the sidebar
path = "Logo.png"
image = Image.open(path)
st.sidebar.image(image, width=300)

# # Hide the rerun button
# st.markdown(""" <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# </style> """, unsafe_allow_html=True)

# Suppression des marges par défaut
# padding = 1
# st.markdown(f""" <style>
#     .reportview-container .main .block-container{{
#         padding-top: {padding}rem;
#         padding-right: {padding}rem;
#         padding-left: {padding}rem;
#         padding-bottom: {padding}rem;
#     }} </style> """, unsafe_allow_html=True)

# ====================================================================
# Customer selection
# ====================================================================

html_select_client="""
    <div class="card">
      <div class="card-body" style="border-radius: 5px 5px 0px 0px;
                  background: #607091; padding-top: 5px; width: auto;
                  height: 40px;">
        <h3 class="card-title" style="background-color:#607091; color:White;
                   font-family:Georgia; text-align: center; padding: 0px 0;">
          Customer & loan information
        </h3>
      </div>
    </div>
    """

st.markdown(html_select_client, unsafe_allow_html=True)

def main():

    ##################################
    # LIST OF API REQUEST FUNCTIONS

    # Get list of SK_IDS (cached)
    @st.cache
    def get_sk_ids():
        
        # Requesting the API and saving the response
        response = requests.get("https://h7o-fastapi-heroku.herokuapp.com/sk_ids")
#        response = requests.get("http://localhost:8000/sk_ids")
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of SK_IDS from the content
        SK_IDS = pd.Series(content['data']).values
        return SK_IDS
    
    # ------------------------------------------------
    # Select the customer's ID
    # ------------------------------------------------

    SK_IDS = get_sk_ids()
    customer_id = st.sidebar.selectbox('Customer ID:', SK_IDS, key=1)
    st.write('Customer id: ', customer_id)
    
      
    # Get basic info from customer (cached)
    @st.cache
    def cust_info(customer_id):
        
        # Requesting the API and saving the response
        response = requests.get("https://h7o-fastapi-heroku.herokuapp.com/customer_info/?SK_ID_CURR=" + str(customer_id))
#        response = requests.get("http://localhost:8000/customer_info/?SK_ID_CURR=" + str(customer_id))
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        df = pd.DataFrame.from_dict(content, orient="index")
        dataframe = df.T
        # dataframe.reset_index(drop=True, inplace=True)
        return dataframe
 
    # Loan information
    # Get basic loan info from customer (cached)
    @st.cache
    def cust_loan(customer_id):
        
        # Requesting the API and saving the response
        response = requests.get("https://h7o-fastapi-heroku.herokuapp.com/customer_loan/?SK_ID_CURR=" + str(customer_id))
#        response = requests.get("http://localhost:8000/customer_loan/?SK_ID_CURR=" + str(customer_id))
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        df = round(pd.DataFrame.from_dict(content, orient="index"), 2)
        dataframe = df.T
        
        return dataframe

    st.table(cust_info(customer_id))
    
    st.table(cust_loan(customer_id))
    
    # Get all original info from customer (cached)
    @st.cache
    def all_info(customer_id):
        
        # Requesting the API and saving the response
        
        response = requests.get("https://h7o-fastapi-heroku.herokuapp.com/origin_data/?SK_ID_CURR=" + str(customer_id))    
#        response = requests.get("http://localhost:8000/origin_data/?SK_ID_CURR=" + str(customer_id))
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        df = pd.DataFrame.from_dict(content, orient="index")
        dataframe = df.T
        test = dataframe.astype(str)
        
        return test
    
    
    # ====================================================================
    # SCORE - PREDICTIONS
    # ====================================================================

    html_score="""
        <div class="card">
          <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                      background: #607091; padding-top: 5px; width: auto;
                      height: 50px;">
            <h3 class="card-title" style="background-color:#607091; color:White;
                       font-family:Georgia; text-align: center; padding: 0px 0;">
              Credit Score 
            </h3>
          </div>
        </div>
        """

    st.markdown(html_score, unsafe_allow_html=True)

    
    
#    Get Personal data (cached)
    @st.cache
    def get_customer_data(customer_id):
        
        # save the response to API request
        response = requests.get("https://h7o-fastapi-heroku.herokuapp.com/customer_data/?SK_ID_CURR="+ str(customer_id))
#        response = requests.get("http://localhost:8000/customer_data/?SK_ID_CURR="+ str(customer_id))
        # convert from JSON format to Python dict
        content = json.loads(response.content)
        # convert data to pd.Series
        cust_pers_data = content['data']
        return cust_pers_data

    cust_pers_data = get_customer_data(customer_id)

 # Get data to display in the gauge

    def get_cust_scoring(customer_id):
              
        # Requesting the API and save the response
        response = requests.post("https://h7o-fastapi-heroku.herokuapp.com/scoring", json=cust_pers_data)
#        response = requests.post("http://localhost:8000/scoring", json=cust_pers_data)
        # convert from JSON format to Python dict
        content = response.json()
        # getting the values from the content
        score = content['probability']
        score = int(np.rint(score * 100))
        
        return score
    
    score = get_cust_scoring(customer_id)       
    
    # Get Personal data (cached)
    @st.cache
    def get_shap_data(customer_id):
        
        # save the response to API request
        response = requests.get("https://h7o-fastapi-heroku.herokuapp.com/shap_data/?SK_ID_CURR="+ str(customer_id))
#        response = requests.get("http://localhost:8000/shap_data/?SK_ID_CURR="+ str(customer_id))
        # convert from JSON format to Python dict
        content = json.loads(response.content)
        # convert data to pd.Series
        cust_shap_data = content
        return cust_shap_data

    shap_data_customer = get_shap_data(customer_id)
    
    # ============== 10 closest neighbors customer score =============
    mean_neigh_score = int(np.rint(dash_df[
    dash_df['SK_ID_CURR'] == customer_id]['10_NEIGH_MEAN_SCORE'] * 100))
    
    # Credit score Gauge plot ==========================================
    
    fig_jauge = go.Figure(go.Indicator(
    mode = 'gauge+number+delta',
    
    value = score,  
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': 'Customer Credit Score (% to default)', 'font': {'size': 20}},
    delta = {'reference': mean_neigh_score,
             'increasing': {'color': 'Crimson'},
             'decreasing': {'color': 'Green'}},
    gauge = {'axis': {'range': [None, 40],
                      'tickwidth': 5,
                      'tickcolor': 'black'},
             'bar': {'color': 'white', 'thickness' : 0.25},
             'bgcolor': 'black',
             'borderwidth': 2,
             'bordercolor': 'gray',
             'steps': [{'range': [0, 5], 'color': 'Green'},
                       {'range': [6, 10], 'color': 'LimeGreen'},
                      # {'range': [49.5, 50.5], 'color': 'red'},
                       {'range': [11, 15], 'color': 'Orange'},
                       {'range': [16, 100], 'color': 'red'}],
             'threshold': {'line': {'color': 'white', 'width': 10},
                           'thickness': 1,
                           'value': score}}))

    fig_jauge.update_layout(paper_bgcolor='white',
                            height=300, width=400,
                            font={'color': 'darkblue', 'family': 'Arial'},
                            margin=dict(l=0, r=0, b=0, t=0, pad=0))

    with st.container():
        # JAUGE 
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.plotly_chart(fig_jauge)
        with col2:
            st.write("")
#            st.write("")
#            st.write("")
#            st.write("")
#            st.write("")
            # Text for the gauge
            if 0 <= score < 5:
                score_text = 'Credit score : EXCELLENT, Loan ACCEPTED'
                st.success(score_text)
            elif 25 <= score < 10:
                score_text = 'Credit score : GOOD, Loan ACCEPTED'
                st.success(score_text)
            elif 50 <= score < 15:
                score_text = 'Credit score : AVERAGE, Loan REFUSED'
                st.warning(score_text)
            else :
                score_text = 'Credit score : BAD, Loan REFUSED'
                st.error(score_text)
            st.write("")    
            st.markdown(f'Average Credit score of the 10 closest neighbors : **{mean_neigh_score}**')
    
    # --------------------------------------------------------------------
    # GENERAL INFORMATION
    # --------------------------------------------------------------------
    def customer_information():
        ''' Display customers information
        '''
        html_customer_information="""
            <div class="card">
                <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                      background: #607091; padding-top: 5px; width: auto;
                      height: 40px;">
                      <h3 class="card-title" style="background-color:#607091; color:White;
                          font-family:Georgia; text-align: center; padding: 0px 0;">
                          Customer information
                      </h3>
                </div>
            </div>
            """
       
    # ====================== CUSTOMER INFORMATION =========================== 
        if st.sidebar.checkbox("Customer data"):     

            st.markdown(html_customer_information, unsafe_allow_html=True)

            with st.spinner('**Display customer data...**'):                 

                with st.expander('Input data for the selected customer Nbr:' + str(customer_id),
                                 expanded=True):
                        st.dataframe(all_info(customer_id)) 
                    
   
    st.sidebar.subheader('Database')
    customer_information()
    
    # --------------------------------------------------------------------
    # INTERPRETABILITY : SHAP VALUES
    # --------------------------------------------------------------------
    
    def get_features_importance():
        ''' Display features importance
        '''
        html_features_importance="""
            <div class="card">
                <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                      background: #607091; padding-top: 5px; width: auto;
                      height: 40px;">
                      <h3 class="card-title" style="background-color:#607091; color:White;
                          font-family:Georgia; text-align: center; padding: 0px 0;">
                          Features importance
                      </h3>
                </div>
            </div>
            """
    
        # ====================== SHAP VALUES =========================== 
        if st.sidebar.checkbox("Features importance"):     

            st.markdown(html_features_importance, unsafe_allow_html=True)

            with st.spinner('** Display parameters impacting the selected customer...**'):                 

                with st.expander('Features impact for the selected customer Nbr:' + str(customer_id),
                                  expanded=True):
                    
    
                    test_set = data_test.reset_index(drop=True)
                    customer_index = test_set[test_set['SK_ID_CURR'] == customer_id].index.item()
                    X_set = test_set.set_index('SK_ID_CURR')
                    X_test_selected = X_set.iloc[customer_index]

                    col1, col2 = st.columns([1, 1])
                    # Graphical display of SHAP features interpretability for the selected customer
                    with col1:

                        plt.clf()

                        # BarPlot selected customer
                        shap.plots.bar(shap_values[customer_index], max_display=40)


                        fig = plt.gcf()
                        fig.set_size_inches((10, 20))
                        # Plot the graph on the dashboard
                        st.pyplot(fig)

                    # Decision plot for the selected customer
                    with col2:
                        plt.clf()

                        # Décision Plot
                        shap.decision_plot(exp_value, np.array(shap_data_customer['1']),
                                           X_test_selected)

                        fig = plt.gcf()
                        fig.set_size_inches((10, 15))
                        # Plot the graph on the dashboard
                        st.pyplot(fig)

    st.sidebar.subheader('Features importance')
    get_features_importance()
  
    
if __name__ == '__main__':
    main()