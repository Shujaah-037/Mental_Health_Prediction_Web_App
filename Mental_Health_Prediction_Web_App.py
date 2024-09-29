import numpy as np
import joblib
import streamlit as st
from PIL import Image  # Import for loading the image

# Loading the saved model in binary mode
loaded_model = joblib.load(open('kmeans_model1.pkl', 'rb'))

# Define the path to the cluster image
image_path = r'C:\Users\shuja\.spyder-py3\Deploy Student Mental Health Cluster\cluster.png'

def mental_health_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Perform the prediction
    prediction = loaded_model.predict(input_data_reshaped)
    
    # Determine the cluster result
    if prediction[0] == 0:
        return 'The student belongs to Cluster 0'
    elif prediction[0] == 1:
        return 'The student belongs to Cluster 1'
    elif prediction[0] == 2:
        return 'The student belongs to Cluster 2'
    elif prediction[0] == 3:
        return 'The student belongs to Cluster 3'
    elif prediction[0] == 4:
        return 'The student belongs to Cluster 4'
    else:
        return 'The student belongs to Cluster 5'

def main():
    # giving a title
    st.title('Mental Health Prediction Web App')

    # getting the input data from the user
    try:
        Fjob_services = float(st.text_input('Father’s job (Services) (e.g., -1.0 to 2.0)', value=0))
        Fjob_other = float(st.text_input('Father’s job (Other) (e.g., -1.0 to 1.0)', value=0))
        guardian_other = float(st.text_input('Guardian (Other) (e.g., -1.0 to 4.0)', value=0))
        Medu = float(st.text_input('Mother’s education (e.g., -2.0 to 2.0)', value=0))
        address_U = float(st.text_input('Address (Urban) (e.g., -2.0 to 0.6)', value=0))
        goout = float(st.text_input('Social interaction (e.g., -1.5 to 1.5)', value=0))
        G2 = float(st.text_input('Second Grade (e.g., -4.0 to 2.0)', value=0))
    except ValueError:
        st.error("Please enter valid numerical values.")

    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Mental Health Result'):
        diagnosis = mental_health_prediction([Fjob_services, Fjob_other, guardian_other, Medu, address_U, goout, G2])
        st.success(diagnosis)
        
        # Display the cluster image
        cluster_image = Image.open(image_path)
        st.image(cluster_image, caption='Cluster Visualization', use_column_width=True)

if __name__ == '__main__':
    main()
