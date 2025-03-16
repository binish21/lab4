from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("fish_market_model.pkl")

@app.route('/')
def home():
    # Render the homepage with the input form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        species = float(request.form['species'])
        length1 = float(request.form['length1'])
        length2 = float(request.form['length2'])
        length3 = float(request.form['length3'])
        height = float(request.form['height'])
        width = float(request.form['width'])

        # Create a numpy array for the input features
        input_data = np.array([[species, length1, length2, length3, height, width]])

        # Make a prediction using the model
        prediction = model.predict(input_data)

        # Return the prediction to the user
        return render_template('index.html', prediction_text=f'Predicted Weight: {prediction[0]:.2f} grams')
    except Exception as e:
        # Handle errors gracefully
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)