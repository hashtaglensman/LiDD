# app.py
from opt_inf_pip import pred_cred
from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
# import pandas as pd
# import numpy as np
# Initalise the Flask app
app = Flask(__name__)

# Loads pre-trained model
# model = load_model('deployment_28042020')

# cols = ['video .mp4']

@app.route('/')
def home():
    return render_template("templates/home.html")

@app.route('/predict',methods=['POST'])
def predict(file):
    fi_n = file
    # int_features = [x for x in request.form.values()]
    # final = np.array(int_features)
    # data_unseen = pd.DataFrame([final], columns = cols)
    prediction = pred_cred(fi_n)
    # prediction = int(prediction.Label[0])
    if prediction == 0:
        pred = 'Real'
    else:
        pred = 'Fake'
    return render_template('templates/home.html',pred=f'The content may be {pred}')

@app.route('/predict_api',methods=['POST'])
def predict_api(file):
    data = request.get_json(force=True)
    data_unseen = file #pd.DataFrame([data])
    prediction = pred_cred(data_unseen)#predict_model(model, data=data_unseen)
    # output = prediction.Label[0]
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)