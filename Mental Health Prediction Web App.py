# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 16:57:08 2024

@author: shuja
"""

import numpy as np
import joblib
import streamlit as st

# Loading the saved model in binary mode
loaded_model = joblib.load(open('kmeans_model1.pkl', 'rb'))


# Top 7 features (example feature names)
important_features = ['Fjob_services', 'Fjob_other', 'guardian_other', 'Medu', 'address_U', 'goout', 'G2']

# Let user input the 7 key features
st.title('Mental Health Status Prediction')
user_input = {}
for feature in important_features:
    user_input[feature] = st.slider(f"Input your {feature}:", min_value=0, max_value=10, value=5)

# Convert user input into a numpy array for the clustering model
user_data = np.array([list(user_input.values())])

# Assuming 'kmeans_model' is your pre-trained clustering model
user_cluster = loaded_model.predict(user_data)

st.write(f'The predicted cluster for this student is: Cluster {user_cluster[0]}')
