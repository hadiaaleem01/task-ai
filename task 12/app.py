# app.py - FINAL WORKING VERSION
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained LinearRegression model
model = pickle.load(open('lr_model.pkl', 'rb'))

# YE HAIN WO 8 FEATURES JO TERE MODEL NE DEKHE HAIN TRAINING ME
feature_names = [
    'Date', 'Store ID', 'Product ID', 'Inventory Level',
    'Demand Forecast', 'Weather Condition', 'Holiday/Promotion',
    'Competitor Pricing'
    # Seasonality ko hata diya kyunki tere processed_data.csv mein nahi hai!
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []
        for feature in feature_names:
            val = request.form.get(feature)
            if val == "" or val is None:
                return render_template('index.html', prediction_text="Error: Sab fields bharo!")
            input_data.append(float(val))

        # Ab exactly 8 features jaayenge
        final_input = np.array(input_data).reshape(1, -1)
        prediction = model.predict(final_input)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', 
                             prediction_text=f"Predicted Price: ₹{prediction}",
                             success=True)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)