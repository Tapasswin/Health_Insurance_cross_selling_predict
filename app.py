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


@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/predict', methods=['POST'])
def predict():
    data=[x for x in request.form.values()]
    # processed_data = data_preprocessing(data)
    # Robust Scalar
    new_data = robust.transform(np.array(data).reshape(1, -1))
    output = dt_mod_ht.predict(new_data)
    if output[0] == 0:
        r = "Not Interested in Vehicle Insurance"
    else:
        r = "Interested in Vehicle Insurance"
    return render_template("home.html", prediction_text=r)


if __name__ == "__main__":
    app.run(debug=True)
