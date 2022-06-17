import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
# Create flask app
flask_app = Flask(__name__)
loaded_model = pickle.load(open("calories_prediction_model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST","GET"])
def predict():
    feature = [float(x) for x in request.form.values()]
    features = [np.array(feature)]
    prediction = loaded_model.predict(features)
    return render_template("index.html", prediction_text = "Number of calories could be burnt are {:.2f}".format(prediction[0]))

@flask_app.route("/reset")
def reset():
    return render_template("index.html")
if __name__ == "__main__":
    flask_app.run(debug=True)
