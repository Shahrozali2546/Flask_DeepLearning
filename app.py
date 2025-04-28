from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, request, render_template

# Load the pre-trained model
model = load_model('model.keras')  

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        int_features = [
            request.form['CreditScore'],
            request.form['Geography'],
            request.form['Gender'],
            request.form['Age'],
            request.form['Tenure'],
            request.form['Balance'],
            request.form['NumOfProducts'],
            request.form['HasCrCard'],
            request.form['IsActiveMember'],
            request.form['EstimatedSalary']
        ]
        
        # Handle encoding for Geography and Gender
        geography = int_features[1].strip().lower()
        gender = int_features[2].strip().lower()

        geo_France, geo_Germany, geo_Spain = 0, 0, 0
        if geography == 'france':
            geo_France = 1
        elif geography == 'germany':
            geo_Germany = 1
        elif geography == 'spain':
            geo_Spain = 1

        gender_male = 1 if gender == 'male' else 0

        # Convert features to float and handle encoding (adjusted to only 10 features)
        final_features = np.array([[  
            float(int_features[0]),  # CreditScore
            float(int_features[3]),  # Age
            float(int_features[4]),  # Tenure
            float(int_features[5]),  # Balance
            float(int_features[6]),  # NumOfProducts
            float(int_features[7]),  # HasCrCard
            float(int_features[8]),  # IsActiveMember
            float(int_features[9]),  # EstimatedSalary
            geo_France,              # Geo France
            geo_Germany               # Geo Germany
            # Note: Removed geo_Spain and gender_male to match the model's expected input shape
        ]])

        # Make prediction
        prediction = model.predict(final_features)

        # Output prediction (classification threshold > 0.5)
        output = (prediction > 0.5).astype(int)
        final_output = int(output[0][0])  # Final output for classification

        # Return prediction to the frontend
        return render_template('index.html', prediction_text=f'Predicted Class: {final_output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
