import numpy as np
import joblib

# Loading the saved model in binary mode
loaded_model = joblib.load(open('kmeans_model1.pkl', 'rb'))

# Example input data
input_data = (-0.6, -1.13, -0.27, 1.24, 0.61, 0.71, -1.57)

# Changing the input_data to a NumPy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make the prediction
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)


if (prediction[0] == 0):
  print('The student belongs to Cluster 0')
elif (prediction[0] == 1):
  print('The student belongs to Cluster 1')
elif (prediction[0] == 2):
  print('The student belongs to Cluster 2')
elif (prediction[0] == 3):
  print('The student belongs to Cluster 3')
elif (prediction[0] == 4):
  print('The student belongs to Cluster 4')
else:
  print('The student belongfs to Cluster 5')