from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("rf_acc_68.pkl")
scaler = joblib.load("normalizer.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inner-page')
def inner_page():
    return render_template('inner-page.html')

@app.route('/portfolio-details')
def portfolio_details():
    return render_template('portfolio-details.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect and convert form data
        features = [
            float(request.form['Age']),
            int(request.form['Gender']),
            float(request.form['Total_Bilirubin']),
            float(request.form['Direct_Bilirubin']),
            float(request.form['Alkaline_Phosphotase']),
            float(request.form['SGPT']),
            float(request.form['SGOT']),
            float(request.form['Total_Protiens']),
            float(request.form['Albumin']),
            float(request.form['Albumin_and_Globulin_Ratio'])
        ]

        # Scale and predict
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
        print("Predicted value:", prediction)


        # Generate result message
        result = "Patient is likely to have Liver Cirrhosis." if prediction == 1 else "Patient is NOT likely to have Liver Cirrhosis."
        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

