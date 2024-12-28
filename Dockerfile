# Use an official Python runtime as a base image
FROM tensorflow/tensorflow:2.11.0-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5500 to allow communication with the Flask app
EXPOSE 5500

# Define the command to run the Flask app
CMD ["python", "app.py"]
