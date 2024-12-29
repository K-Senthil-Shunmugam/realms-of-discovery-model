
# Realms of Discovery - Command Parser Model

## Overview

This repository contains a machine learning model for parsing commands in a specific domain, built using deep learning techniques. The project includes the necessary scripts, model files, and configuration for training, deploying, and testing the model.

## Features

- **Deep Learning Model**: Pre-trained model (`command_parser_model.h5`) for parsing commands.
- **Data Generation**: `generate_data.py` for creating or preprocessing training data.
- **Training Notebook**: Jupyter notebook (`train-model.ipynb`) for training the model from scratch.
- **Deployment Ready**: Dockerfile and requirements for seamless deployment.
- **CI/CD**: GitLab CI/CD pipeline configuration (`.gitlab-ci.yml`).

## File Structure

- **`app.py`**: Main application script for running the model or serving it via an API.
- **`generate_data.py`**: Script for data preparation.
- **`train-model.ipynb`**: Notebook for model training.
- **`command_parser_model.h5`**: Pre-trained model file.
- **`label_encoder.pkl`**: Encodes and decodes class labels.
- **`tokenizer.pkl`**: Tokenizer for text preprocessing.
- **`requirements.txt`**: List of Python dependencies.
- **`Dockerfile`**: Instructions for containerizing the application.
- **`.gitlab-ci.yml`**: Configuration for GitLab CI/CD.

## Prerequisites

- Python 3.7+
- Docker (optional, for containerization)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/realms-of-discovery.git
   cd realms-of-discovery
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Build Docker container:
   ```bash
   docker build -t command-parser .
   ```

## Usage

### Running the Application
Run the application locally:
```bash
python app.py
```

### Training the Model
Open `train-model.ipynb` in Jupyter Notebook to train the model.

### Generating Data
Use `generate_data.py` to preprocess data:
```bash
python generate_data.py
```
