import os
from flask import Flask, request, jsonify
import ImageUtils
import RecogitionService
import numpy as np

app = Flask("recognition-service")

@app.route('/features', methods=['POST'])
def handle_extract_face_features():
    image = ImageUtils.decode_image_buffer(bytearray(request.json['face']['data']))
    grayscale_image = ImageUtils.rgb_to_grayscale(image)
    normalized_image = ImageUtils.normalize(grayscale_image)
    aligned_face = ImageUtils.grayscale_to_rgb(RecogitionService.align_face(normalized_image))
    face_features = RecogitionService.extract_features(aligned_face)

    return jsonify({"features": list(face_features)})

@app.route('/recognize', methods=['POST'])
def handle_recognize_face_from_features():
    features = np.array(request.json['features'])
    face_to_recognize = ImageUtils.decode_image_buffer(bytearray(request.json['face']['data']))
    grayscale_image = ImageUtils.rgb_to_grayscale(face_to_recognize)
    normalized_image = ImageUtils.normalize(grayscale_image)
    aligned_face = ImageUtils.grayscale_to_rgb(RecogitionService.align_face(normalized_image))
    face_to_recognize_features = RecogitionService.extract_features(aligned_face)
    are_same = RecogitionService.recognize(features, face_to_recognize_features)

    return jsonify({"areSame": are_same})

app.run(port=4003)
