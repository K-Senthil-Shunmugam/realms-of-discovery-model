from flask import Flask, request, jsonify
import mlflow
import mlflow.keras
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import redis
import hashlib
import json

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

# Set up Redis connection
redis_client = redis.StrictRedis(host='192.168.192.25', port=6379, db=0, decode_responses=True)

# Function to generate a cache key based on command
def generate_cache_key(command):
    return hashlib.md5(command.encode('utf-8')).hexdigest()

# Function for prediction
def predict_command(command, threshold=0.5):
    # Convert the command to sequence of integers
    sequence = tokenizer.texts_to_sequences([command])
    # Pad the sequence to the specified length
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=5)
    
    # Predict the class probabilities
    prediction = model.predict(padded_sequence)
    
    # Get the class probabilities for all classes
    probabilities = prediction[0]
    
    # Check if the maximum probability is below the threshold
    if np.max(probabilities) < threshold:
        return command  # Return the input if probability is below the threshold
    
    # Otherwise, return the label of the predicted class
    predicted_label = np.argmax(probabilities)
    return label_encoder.inverse_transform([predicted_label])[0]

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input command from JSON request
        data = request.get_json()
        command = data['command']
        
        # Check if the result is in the cache
        cache_key = generate_cache_key(command)
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            # Return the cached result
            return jsonify({"command": cached_result, "source": "cache"}), 200
        else:
            # Make prediction
            predicted_label = predict_command(command)
            
            # Store the result in the cache for future use
            redis_client.set(cache_key, predicted_label, ex=3600)  # Expiry set to 1 hour
            
            return jsonify({"command": predicted_label, "source": "model"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Run the Flask app on port 5500
    app.run(host="0.0.0.0", port=5500)
