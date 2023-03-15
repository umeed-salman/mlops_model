# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
app = Flask(__name__)
# Load the model
model = pickle.load(open('finalized_model.pkl','rb'))
@app.route('/api/<exp>')
def predict(exp):
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([[float(exp)]])
    # Take the first value of prediction
    output = prediction.item()
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
