from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np


app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
@app.route('/')

def index():
    """
    The function sorts unique locations from a data variable and passes them as a parameter to an HTML
    template.
    
    Returns:
      The function `index()` is returning a rendered HTML template called 'index.html' with a list of
    unique locations sorted in alphabetical order as a variable called `locations`.
"""
    locations =sorted(data['location'].unique())
    return render_template('index.html',locations =locations)

pipe = joblib.load('./Ridge.pkl')
@app.route('/predict', methods=['POST'])

def predict():
    """
    The function takes input values for location, bhk, bathroom, and sqft, creates a dataframe with the
    input values, and returns a prediction of the house price using a pre-trained machine learning
    model.
    
    Returns:
      a string representation of the predicted price of a house based on the input values of location,
    number of bedrooms, number of bathrooms, and total square footage. The predicted price is calculated
    using a machine learning model stored in a pipeline object called "pipe". The predicted price is
    rounded to two decimal places using the numpy library.
    """
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bathroom')
    sqft = request.form.get('sqft')
    #print(location,bhk,bath,sqft)
    input = pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
    prediction = pipe.predict(input)[0]*100000
    
    return str(np.round(prediction,2))


if __name__=="__main__":
    app.run(debug=True, port=5001)