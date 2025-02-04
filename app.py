from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('voting_clf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input values from form
    features = [
        float(request.form['alcohol']),
        float(request.form['malic_acid']),
        float(request.form['ash']),
        float(request.form['alcalinity_of_ash']),
        float(request.form['magnesium']),
        float(request.form['total_phenols']),
        float(request.form['flavanoids']),
        float(request.form['nonflavanoid_phenols']),
        float(request.form['proanthocyanins']),
        float(request.form['color_intensity']),
        float(request.form['hue']),
        float(request.form['od280/od315_of_diluted_wines']),
        float(request.form['proline'])
    ]

    # Convert to NumPy array and reshape for model input
    input_data = np.array([features])

    # Make prediction
    prediction = model.predict(input_data)
    prediction_text = f"Predicted Wine Class: {prediction[0]}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)  # Change to False in production


