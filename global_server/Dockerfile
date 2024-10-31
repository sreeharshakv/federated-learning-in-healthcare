# Use a base image with Python
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install pkg-config and other dependencies needed to build Python packages
RUN apt-get update && apt-get install -y \
    pkg-config \
    build-essential \
    libhdf5-dev

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Run the Flask app
CMD ["python", "global_server.py"]
