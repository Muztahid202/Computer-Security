from flask import Flask, send_from_directory, request, jsonify
# additional imports
import os
import uuid
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

stored_traces = []
stored_heatmaps = []

# Ensure heatmap directory exists
os.makedirs('static/heatmap', exist_ok=True)

class FingerprintClassifier(nn.Module):
    """Basic neural network model for website fingerprinting classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(FingerprintClassifier, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After two 2x pooling operations
        self.fc_input_size = conv_output_size * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
class ComplexFingerprintClassifier(nn.Module):
    """A more complex neural network model for website fingerprinting classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(ComplexFingerprintClassifier, self).__init__()
        
        # 1D Convolutional layers with batch normalization
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After three 2x pooling operations
        self.fc_input_size = conv_output_size * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size*2)
        self.bn4 = nn.BatchNorm1d(hidden_size*2)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
# Model setup
INPUT_SIZE = 1000
HIDDEN_SIZE = 128
NUM_CLASSES = 3  # moodle, google, prothomalo

model = ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
model.load_state_dict(torch.load("saved_models/complex_cnn.pth", map_location=torch.device('cpu')))
model.eval()

# Label mapping
INDEX_TO_WEBSITE = {
    0: "https://cse.buet.ac.bd/moodle/",
    1: "https://google.com",
    2: "https://prothomalo.com"
}

def normalize_trace(trace, length=1000):
    if not trace:
        return [0.0] * length
    min_val = min(trace)
    max_val = max(trace)
    if max_val == min_val:
        normalized = [0.0] * len(trace)
    else:
        normalized = [(x - min_val) / (max_val - min_val) for x in trace]
    return normalized[:length] + [0.0] * (length - len(normalized))

    

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/collect_trace', methods=['POST'])
def collect_trace():
    """ 
    Implement the collect_trace endpoint to receive trace data from the frontend and generate a heatmap.
    1. Receive trace data from the frontend as JSON
    2. Generate a heatmap using matplotlib
    3. Store the heatmap and trace data in the backend temporarily
    4. Return the heatmap image and optionally other statistics to the frontend
    """
    try:
        trace_data = request.json.get('trace')
        if not trace_data:
            return jsonify({'error': 'No trace data provided'}), 400

        # Calculate statistics
        min_val = min(trace_data)
        max_val = max(trace_data)
        trace_range = max_val - min_val
        sample_count = len(trace_data)

        # Generate heatmap
        plt.figure(figsize=(10, 6))
        plt.imshow([trace_data], aspect='auto', cmap='hot', interpolation='nearest')
        plt.colorbar(label='Access Count')
        plt.title('Cache Access Pattern Heatmap')
        plt.xlabel('Time Window')
        plt.ylabel('Trace')

        # Save heatmap
        heatmap_filename = f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
        heatmap_path = os.path.join('static', 'heatmap', heatmap_filename)
        plt.savefig(heatmap_path)
        plt.close()

        # Store data
        stored_traces.append(trace_data)
        stored_heatmaps.append(heatmap_filename)
        
         # Normalize and convert to tensor
        normalized_trace = normalize_trace(trace_data)
        input_tensor = torch.tensor([normalized_trace], dtype=torch.float32)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            predicted_index = torch.argmax(output, dim=1).item()
            predicted_website = INDEX_TO_WEBSITE[predicted_index]

        return jsonify({
            'heatmap': f'/heatmap/{heatmap_filename}',
            'message': 'Trace collected successfully',
            'stats': {
                'min': min_val,
                'max': max_val,
                'range': trace_range,
                'samples': sample_count
            },
            'prediction': predicted_website

        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/get_results', methods=['GET'])
def get_results():
    """ Return the currently stored traces as JSON """
    try:
        return jsonify({
            'traces': stored_traces,
            'collectedAt': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_results', methods=['POST'])
def clear_results():
    """ 
    Implment a clear results endpoint to reset stored data.
    1. Clear stored traces and heatmaps
    2. Return success/error message
    """
    try:
        # Clear stored data
        stored_traces.clear()
        stored_heatmaps.clear()
        
        # Clear heatmap files
        for filename in os.listdir('static/heatmap'):
            file_path = os.path.join('static/heatmap', filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                
        return jsonify({'message': 'All results cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Additional endpoints can be implemented here as needed.

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)