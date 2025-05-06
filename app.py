import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# This will load  the trained Random forest model
model = joblib.load("models/random_forest_model.pkl")


# Defining all the pages 
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        try:
            age = int(request.form["age"])
            gender = request.form["gender"]
            education = request.form["education"]
            experience = float(request.form["experience"])

# Convert the inputs to numeric values 
            gender_encoded = 1 if gender == "Male" else 0
            education_map = {"Bachelor's": 0, "Master's": 1, "PhD": 2}
            education_encoded = education_map[education]

# Prepare input for prediction
            input_data = np.array([[age, gender_encoded, education_encoded, experience]])
            predicted_salary = model.predict(input_data)[0]
            prediction = f"£{int(predicted_salary):,}"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("predict.html", prediction=prediction)

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')



@app.route("/about")
def about():
    return render_template("about.html")

# Debug mode makes it reload automatically
if __name__ == "__main__":
    app.run(debug=True)
