from flask import Flask, request, jsonify
from model import build_model, aggregate_weights
import numpy as np

app = Flask(__name__)

# Global model and weight storage
global_model = build_model()
node_weights = {}  # Dictionary to store the latest weights for each node
expected_nodes = ["node_1", "node_2"]  # List of all expected nodes

@app.route('/submit_weights', methods=['POST'])
def receive_weights():
    node_id = request.json['node_id']
    received_weights = [np.array(w) for w in request.json['weights']]

    # Store or update weights for this node
    node_weights[node_id] = received_weights

    # Check if all nodes have submitted at least once
    if len(node_weights) == len(expected_nodes):
        # Aggregate weights using the latest weights from each node
        all_weights = list(node_weights.values())
        aggregated_weights = aggregate_weights(all_weights)
        global_model.set_weights(aggregated_weights)
        aggregation_complete = True
    else:
        aggregation_complete = False

    return jsonify({"status": "success", "message": f"Weights received for {node_id}"}), 200

@app.route('/get_global_weights', methods=['GET'])
def send_global_weights():
    # Check if initial aggregation was completed at least once
    if len(node_weights) < len(expected_nodes):
        return jsonify({"status": "error", "message": "Initial aggregation incomplete. Waiting for all nodes."}), 400

    # Return aggregated global weights
    global_weights = [w.tolist() for w in global_model.get_weights()]
    return jsonify({"status": "success", "global_weights": global_weights}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)