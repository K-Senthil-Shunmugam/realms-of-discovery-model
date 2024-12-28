from flask import Flask, request, jsonify
import mlflow.pyfunc
import tensorflow as tf
import numpy as np
import json
from mlflow.tracking import MlflowClient

# Initialize the Flask app
app = Flask(__name__)

# Set up MLflow tracking URI
mlflow.set_tracking_uri("http://192.168.192.25:5600")

# Fetch the run ID for the specified run name
try:
    client = MlflowClient()
    experiment_id = client.get_experiment_by_name("seq2seq-model").experiment_id

    # List all runs in the experiment
    runs = client.search_runs(experiment_ids=experiment_id, filter_string="tags.mlflow.runName = 'v1'")
    if runs:
        run_id = runs[0].info.run_id
        print(f"Run ID: {run_id}")
    else:
        raise ValueError("Run with the specified name not found.")

    # Load the MLflow model and tokenizer artifacts
    artifact_path = "seq2seq_model"  # Replace with the logged model artifact path
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model loaded successfully from {model_uri}")

    # Load tokenizer from MLflow artifacts
    tokenizer_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="tokenizer/tokenizer.json")
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
    input_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data['input_tokenizer'])
    output_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data['output_tokenizer'])
    print("Tokenizers loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizers: {e}")
    model = None
    input_tokenizer = None
    output_tokenizer = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or input_tokenizer is None or output_tokenizer is None:

        return jsonify({"error": "Model or tokenizers are not loaded."}), 500

    try:
        # Parse input JSON
        input_data = request.get_json()
        input_texts = input_data.get("input_texts", [])

        # Preprocess input texts
        input_sequences = input_tokenizer.texts_to_sequences(input_texts)
        max_encoder_seq_length = 100
        encoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding='post')

        # Perform inference
        predictions = model.predict([encoder_input_data])
        output_sequences = np.argmax(predictions, axis=-1)

        # Convert predicted sequences back to text
        output_texts = []
        for sequence in output_sequences:
            tokens = [output_tokenizer.index_word.get(token, '') for token in sequence if token != 0]
            output_texts.append(''.join(tokens).replace('<end>', ''))

        return jsonify({"predictions": output_texts})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)
