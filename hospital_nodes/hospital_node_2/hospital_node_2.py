import requests
import numpy as np
import pandas as pd
from model import build_model, train_local_model, evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

# Unique identifier for this hospital node
NODE_ID = "node_2"  # Use a unique ID for this node

# Replace with the actual IP address of the global server
GLOBAL_SERVER_URL = 'http://global_server:5000'

# Load local data for Hospital Node 2
data = pd.read_csv('hospital_data_2.csv')

# Define columns to drop that are not needed for prediction
columns_to_drop = ['race', 'gender', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult']
data = data.drop(columns=columns_to_drop)
data['readmitted'] = data['readmitted'].apply(lambda x: 0 if x.lower() == 'no' else 1)

# Encode categorical columns
categorical_columns = ['age', 'insulin', 'change', 'diabetesMed']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# Define features and target
X = data.drop(columns=['readmitted']).values  # Features
y = data['readmitted'].values  # Target: readmission (binary)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure data is in the correct format for TensorFlow
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
y_train = y_train.astype('int32')
y_val = y_val.astype('int32')

# Initialize local model
local_model = build_model(input_shape=(X_train.shape[1],))
rounds = 5  # Number of aggregation rounds
local_epochs = 3  # Local epochs per round
max_retries = 5  # Maximum number of retries for `get_global_weights`
retry_delay = 5  # Delay between retries in seconds

for round_num in range(rounds):
    print(f"\n--- Round {round_num + 1} ---")

    # Train the local model for a few epochs
    local_weights = train_local_model(X_train, y_train, local_model)

    # Evaluate and print local metrics before aggregation
    metrics_before = evaluate_model(local_model, X_val, y_val)
    print(f"Hospital Node {NODE_ID} Metrics before global aggregation (Round {round_num + 1}): {metrics_before}")

    # Prepare payload to send to the global server
    payload = {
        "node_id": NODE_ID,
        "weights": [w.tolist() for w in local_weights]
    }

    # Submit weights to the global server
    response = requests.post(f'{GLOBAL_SERVER_URL}/submit_weights', json=payload)
    if response.status_code == 200:
        print(f"Hospital Node {NODE_ID} submitted weights for Round {round_num + 1}")

        # Retry loop for `get_global_weights` API call
        for attempt in range(max_retries):
            global_weights_response = requests.get(f'{GLOBAL_SERVER_URL}/get_global_weights')
            if global_weights_response.status_code == 200:
                # Successfully received the global weights
                global_weights = [np.array(w) for w in global_weights_response.json()['global_weights']]

                # Update local model with global weights
                local_weights_before = local_model.get_weights()
                local_model.set_weights(global_weights)

                # Calculate weight differences for debugging
                weight_differences = [np.linalg.norm(lw - gw) for lw, gw in zip(local_weights_before, global_weights)]
                print("Weight differences in local model after updating with global weights:", weight_differences)

                # Evaluate and print metrics after aggregation
                metrics_after = evaluate_model(local_model, X_val, y_val)
                print(f"Hospital Node {NODE_ID} Metrics after global aggregation (Round {round_num + 1}): {metrics_after}")
                break
            else:
                print(f"Attempt {attempt + 1}/{max_retries} failed to fetch global weights. Retrying in {retry_delay*(attempt+1)} seconds...")
                time.sleep(retry_delay*(attempt+1))
        else:
            print(f"Failed to fetch global weights after {max_retries} attempts. Skipping update for this round.")
    else:
        print(f"Error submitting weights for Round {round_num + 1}: Status code {response.status_code}")
        print("Response content:", response.content)
