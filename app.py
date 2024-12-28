from flask import Flask, request, jsonify
import mlflow
import mlflow.keras
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set up Flask app
app = Flask(__name__)

# Set the tracking URI for MLflow server
mlflow.set_tracking_uri("http://192.168.192.25:5600")
mlflow.set_experiment("seq2seq-model")

run_id = "16c8c648ebb24b37b88b20c6bc960ce6"
# Load the model from the specified run ID

model_uri = f"runs:/{run_id}/model"
model = mlflow.keras.load_model(model_uri)

# Load the tokenizer and label encoder artifacts
tokenizer_artifact_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/artifacts/tokenizer/tokenizer.pkl")
with open(tokenizer_artifact_path, 'rb') as f:
    tokenizer = pickle.load(f)

label_encoder_artifact_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/artifacts/label_encoder/label_encoder.pkl")
with open(label_encoder_artifact_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Function for prediction
def predict_command(command):
    sequence = tokenizer.texts_to_sequences([command])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=5)  
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction, axis=1)
    return label_encoder.inverse_transform(predicted_label)[0]

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input command from JSON request
        data = request.get_json()
        command = data['command']
        
        # Make prediction
        predicted_label = predict_command(command)
        
        return jsonify({"command": command, "predicted_label": predicted_label}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Run the Flask app on port 5500
    app.run(host="0.0.0.0", port=5500)
