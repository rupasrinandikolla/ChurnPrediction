from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model/churn_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    gender = int(request.form["gender"])
    senior = int(request.form["SeniorCitizen"])
    monthly = float(request.form["MonthlyCharges"])
    total = float(request.form["TotalCharges"])

    # Input data
    data = np.array([[gender, senior, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, monthly, total]])

    # Prediction
    prediction = model.predict(data)

    # Probability
    probability = model.predict_proba(data)

    if prediction[0] == 1:
        result = "Customer Will Churn"
        prob = round(probability[0][1] * 100, 2)
    else:
        result = "Customer Will Stay"
        prob = round(probability[0][0] * 100, 2)

    return render_template(
        "index.html",
        prediction_text=result,
        probability=prob
    )

if __name__ == "__main__":
    app.run(debug=True)