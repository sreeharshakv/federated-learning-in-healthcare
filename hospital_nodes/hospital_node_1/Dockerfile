# Use a base image with Python
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install pkg-config, build-essential, and libhdf5-dev dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    build-essential \
    libhdf5-dev

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Run the hospital node script
CMD ["python", "hospital_node_1.py"]
