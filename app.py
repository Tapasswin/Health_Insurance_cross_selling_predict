import json
import pickle
from flask import Flask
from flask import request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from scipy.stats import stats, zscore

app=Flask(__name__)

# Load the model
dt_mod_ht = pickle.load(open('dt_mod_ht.pkl', 'rb'))
robust = pickle.load(open('robust.pkl', 'rb'))


def data_preprocessing(data):
    # Gender
    if data[0] == "Male":
        data[0] = 1
    else:
        data[0] = 0
    # Vehicle_Age
    if data[5] in ["1-2 Year", "< 1 Year", "> 2 Years"]:
        if data[5] == "1-2 Year":
            data[5] = 0
        elif data[5] == "< 1 Year":
            data[5] = 1
        else:
            data[5] = 2
    else:
        if data[5] > 2:
            data[5] = 2
        elif data[5] < 1:
            data[5] = 1
        else:
            data[5] = 0
    # Vehicle_Damage
    if data[6] == "Yes":
        data[6] = 1
    else:
        data[6] = 0
    return data


@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/predict', methods=['POST'])
def predict():
    data=[x for x in request.form.values()]
    processed_data = data_preprocessing(data)
    # Robust Scalar
    new_data = robust.transform(np.array(processed_data).reshape(1, -1))
    output = dt_mod_ht.predict(new_data)
    if output[0] == 0:
        r = "Not Interested in Vehicle Insurance \n Model Response: {}"
    else:
        r = "Interested in Vehicle Insurance \n Model Response: {}"
    return render_template("home.html", prediction_text=r.format(output[0]))


if __name__ == "__main__":
    app.run(debug=True)
