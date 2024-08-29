from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import sys
from image_data.prepare_data import prepare_data
from image_data.train_test_models import main as image_data_classification
from tabular_data.tabular_data import main as tmain
import image_data.constants as constants

app = Flask(__name__, static_folder='data')

@app.route('/')
def index():
    return render_template('index.html', 
                           image_size=constants.IMAGE_SIZE,
                           slice_mode=constants.SLICE_MODE,
                           test_size=constants.TEST_SIZE,
                           val_size=constants.VAL_SIZE)

@app.route('/run', methods=['POST'])
def run():
    action = request.form['action']
    
    if action == 'image':
        prepare_data()
        image_data_classification()
        return "Image data processing complete"
    elif action == 'tabular':
        tmain()
        return "Tabular data processing complete"
    elif action == 'prepare':
        prepare_data()
        return "Data preparation complete"
    else:
        return "Invalid action"

@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    constants.IMAGE_SIZE = int(data['image_size'])
    constants.SLICE_MODE = data['slice_mode']
    constants.TEST_SIZE = float(data['test_size'])
    constants.VAL_SIZE = float(data['val_size'])
    return jsonify({"status": "success"})

@app.route('/static/data/raw/<path:filename>')
def serve_nifti(filename):
    return send_from_directory('data/raw', filename)

if __name__ == '__main__':
    app.run(debug=True)