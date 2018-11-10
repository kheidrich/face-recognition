import os
from flask import Flask, request, jsonify
import ImageUtils

app = Flask("recognition-service")

@app.route('/train', methods=['POST'])
def handle_train_model():
    print(ImageUtils.to_rgb_array(bytearray(request.json['face']['data'])))
    return jsonify({"model": {"data": [22,22,22,22], "type": "Buffer"}})

@app.route('/recognize', methods=['POST'])
def handle_compare_image_with_model():
    return jsonify({"areSame": True})

app.run(port=4003)
