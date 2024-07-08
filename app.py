from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
try:
    model = joblib.load('model.lb')
    scaler = joblib.load('scaler.lb')
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/data', methods=['GET', 'POST'])
def user_data():
    prediction_text = ""
    if request.method == 'POST':
        try:
            # Collect data from form
            Pregnancies = int(request.form['Pregnancies'])
            Glucose = int(request.form['Glucose'])
            BloodPressure = int(request.form['BloodPressure'])
            SkinThickness = int(request.form['SkinThickness'])
            Insulin = int(request.form['Insulin'])
            BMI = float(request.form['BMI'])
            DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
            Age = int(request.form['Age'])

            # Prepare the data for prediction
            user_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

            
            # Check if the scaler is correctly loaded and can transform data
            if scaler is not None and hasattr(scaler, 'transform'):
                scaled_data = scaler.transform(user_data)
            else:
                raise ValueError("Scaler is not correctly loaded or does not have a 'transform' method")
            
            # Debug: Print the scaled data
            print("Scaled data:", scaled_data)

            # Ensure the model is callable and predict
            if hasattr(model, "predict"):
                prediction = model.predict(scaled_data)[0]
                print("Prediction:", prediction)
                prediction_text = f'Diabetes: {"Yes" if prediction == 1 else "No"}'
            else:
                raise ValueError("The model object does not have a 'predict' method")

        except Exception as e:
            prediction_text = f'Error: {str(e)}'
            print("Error during prediction:", str(e))

    return render_template('data.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
