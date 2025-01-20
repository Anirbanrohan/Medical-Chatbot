from flask import Flask, request, jsonify, render_template, session
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import cohere

# Load the trained model and LabelEncoder
with open('disease_model_rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoder_rf.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Load your disease description and precaution datasets
disease_info_df = pd.read_csv('disease_info.csv')  # Contains descriptions and symptoms
precautions_df = pd.read_csv('disease_precautions.csv')  # Contains disease precautions
final_df = pd.read_csv('final.csv')
symptom_columns = [col for col in final_df.columns if col != 'Disease']

# Initialize Cohere client
co = cohere.Client('zxD60GHuQjAkGnR0HSiUL4OVaPKrmvxIeifXLxFF')  # Replace with your actual API key

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'b3f8d74901a92b3fcb58'  # For session management

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    symptoms = {col: 0 for col in symptom_columns}  # Initialize all symptoms as 0
    
    # Validate and mark the symptoms provided in the request
    input_symptoms = input_data.get('symptoms', [])
    valid_symptoms = [symptom for symptom in input_symptoms if symptom in symptoms]
    
    for symptom in valid_symptoms:
        symptoms[symptom] = 1
    
    input_df = pd.DataFrame([symptoms])
    input_df = input_df[symptom_columns]  # Ensure order matches the model's expectations
    
    predicted_encoded = model.predict(input_df)
    predicted_disease = label_encoder.inverse_transform(predicted_encoded)
    
    probabilities = model.predict_proba(input_df)
    confidence_scores = {disease: prob for disease, prob in zip(label_encoder.classes_, probabilities[0])}
    
    # Save the predicted disease in session for follow-up questions
    session['context'] = {'disease': predicted_disease[0]}
    
    return jsonify({
        'predicted_disease': predicted_disease[0],
        'confidence_scores': confidence_scores
    })

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    context = session.get('context', {})
    
    # Check if the user is continuing a conversation about a disease
    if 'disease' in context:
        disease = context['disease']
        response = co.chat(
            message=f"User asked: {user_message}\nBased on the disease '{disease}', respond appropriately.",
            prompt_truncation='auto'
        )
        answer = response.text
    else:
        # Inform the user that they need to provide symptoms first for disease prediction
        answer = "Please provide symptoms first for disease prediction."
    
    # Update session context
    session['context'] = {
        'disease': context.get('disease', None),  # Maintain current disease if ongoing conversation
    }
    
    return jsonify({'response': answer})


if __name__ == '__main__':
    app.run(debug=True)
