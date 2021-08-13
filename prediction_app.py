#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing packages
import numpy as np
from sklearn import preprocessing
from flask import Flask, request, jsonify, render_template, url_for
import pickle


# In[ ]:


app = Flask(__name__) #creating an instance of Flask class
#loading model
with open('linear_model.pkl', 'rb') as f:
    model = pickle.load(f)


# In[ ]:


@app.route('/')#decorator for the web server to understand
def home():
    return render_template('index.html')


# In[ ]:


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(float(x)) for x in request.form.values()] #forming list of submitted values
    int_features = [np.array(int_features)] #converting the list to numpy array
    prediction = model.predict(int_features)#prediction

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'The Video game Sale is {output}')

if __name__ == "__main__":
    app.run()

