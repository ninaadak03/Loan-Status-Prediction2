from flask import Flask, render_template, request
import pickle
import numpy as np

appp = Flask(__name__)

# Load the model
model = pickle.load(open("rfmodel.pkl", "rb"))

@appp.route('/')
def home():
    return render_template('index2.html')

@appp.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    no_of_dependents = int(request.form['no_of_dependents'])
    education = 1 if request.form['education'] == 'Graduate' else 0
    self_employed = 1 if request.form['self_employed'] == 'Yes' else 0
    income_annum = float(request.form['income_annum'])
    loan_amount = float(request.form['loan_amount'])
    loan_term = float(request.form['loan_term'])
    cibil_score = float(request.form['cibil_score'])
    residential_assets_value = float(request.form['residential_assets_value'])
    commercial_assets_value = float(request.form['commercial_assets_value'])
    luxury_assets_value = float(request.form['luxury_assets_value'])
    bank_asset_value = float(request.form['bank_asset_value'])
    
    # Create feature array for prediction
    features = np.array([[no_of_dependents, education, self_employed, income_annum,
                          loan_amount, loan_term, cibil_score, residential_assets_value,
                          commercial_assets_value, luxury_assets_value, bank_asset_value]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Determine result message
    result = 'Approved' if prediction == 1 else 'Rejected'
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    appp.run(debug=True)
