# -*- coding: utf-8 -*-

from flask import Flask, request
import pickle

app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data
    data = request.get_json()

    # Make a prediction using the trained model
    prediction = model.predict([[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]])

    # Return the predicted class
    return str(prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

