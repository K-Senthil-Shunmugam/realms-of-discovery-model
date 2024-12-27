from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizers
model = tf.keras.models.load_model('seq2seq_model.h5')

with open('source_tokenizer.pkl', 'rb') as f:
    source_tokenizer = pickle.load(f)

with open('target_tokenizer.pkl', 'rb') as f:
    target_tokenizer = pickle.load(f)

# Define maximum lengths
max_source_len = 100  # Adjust as per your model's configuration
max_target_len = 100  # Adjust as per your model's configuration

# Initialize Flask app
app = Flask(__name__)

# Helper function to convert sequences to text
def sequences_to_text(sequences, tokenizer):
    return [' '.join([tokenizer.index_word.get(i, '') for i in seq if i != 0]) for seq in sequences]

# Route to generate predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extract the input text from the request
    source_text = data.get('input_text', '')
    
    # Tokenize the source text
    source_sequence = source_tokenizer.texts_to_sequences([source_text])
    source_sequence = pad_sequences(source_sequence, maxlen=max_source_len, padding='post')
    
    # Prepare the decoder input (empty start token)
    target_sequence = np.zeros((1, max_target_len - 1))

    # Generate prediction
    prediction = model.predict([source_sequence, target_sequence])
    
    # Convert predicted sequence to text
    predicted_sequence = np.argmax(prediction, axis=-1)
    predicted_text = sequences_to_text(predicted_sequence, target_tokenizer)[0]
    
    # Return the predicted text as a JSON response
    return jsonify({'predicted_text': predicted_text})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
