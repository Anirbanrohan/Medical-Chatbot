
### `README.md`

```markdown
# Disease Prediction Web Application

## Overview
This is a web-based disease prediction system that allows users to input symptoms and get a disease prediction based on their input. It also includes a chatbot feature that answers follow-up questions related to the disease.

The project uses machine learning for disease prediction (XGBoost classifier), a Flask backend to handle user inputs, and Cohere API for natural language responses.

## API Endpoints

### 1. `/predict` (POST)
This endpoint predicts the disease based on the symptoms entered by the user.

#### Request:
- **Method**: POST
- **Content-Type**: `application/json`
- **Body**:
```json
{
  "symptoms": ["fever", "cough", "fatigue"]
}
```

#### Response:
- **Status**: 200 OK
- **Body**:
```json
{
  "predicted_disease": "Flu",
  "confidence_scores": {
    "Flu": 0.85,
    "Cold": 0.10,
    "Covid-19": 0.05
  }
}
```

### 2. `/chat` (POST)
This endpoint provides a chatbot interface where users can ask follow-up questions related to the predicted disease. The chatbot responds based on the disease that was predicted during the `/predict` step.

#### Request:
- **Method**: POST
- **Content-Type**: `application/json`
- **Body**:
```json
{
  "message": "What are the precautions for Flu?"
}
```

#### Response:
- **Status**: 200 OK
- **Body**:
```json
{
  "response": "To prevent Flu, make sure to wash your hands regularly and avoid close contact with infected individuals."
}
```

## Deployment Instructions

To deploy the application locally, follow these steps:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/Anirbanrohan/Medical-Chatbot.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Medical-Chatbot
   ```

3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Ensure that you have the model file (`disease_model_rf.pkl`), label encoder file (`label_encoder_rf.pkl`), and the disease description and precautions data files (`disease_info.csv`, `disease_precautions.csv`, `final.csv`) in the project directory.

6. Run the Flask app:
   ```bash
   python app.py
   ```

7. Open a web browser and go to `http://127.0.0.1:5000/` to use the application.

## Usage Guidelines

1. **Disease Prediction**:
   - Enter a list of symptoms in the text field, separated by commas, and click "Submit Symptoms".
   - The application will predict the disease based on the provided symptoms and display the predicted disease along with confidence scores.

2. **Chatbot**:
   - After the disease prediction, you can ask follow-up questions regarding the disease.
   - The chatbot will provide information about the disease based on the prediction.

3. **Error Handling**:
   - If you try to use the chat feature before submitting symptoms, the chatbot will prompt you to provide symptoms first.

## Cohere API Integration

This project uses the [Cohere API](https://cohere.ai/) for generating natural language responses in the chatbot. You will need a valid API key to use this functionality. Make sure to replace the `cohere.Client` initialization with your API key:
```python
co = cohere.Client('<YOUR_API_KEY>')
```
