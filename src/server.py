import os
from flask import Flask, request, jsonify
import ImageUtils
import RecogitionService
import numpy as np
import cv2

app = Flask("recognition-service")

@app.route('/train', methods=['POST'])
def handle_extract_face_features():
    image = ImageUtils.decode_image_buffer(bytearray(request.json['face']['data']))
    grayscale_image = ImageUtils.rgb_to_grayscale(image)
    normalized_image = ImageUtils.grayscale_to_rgb(cv2.resize(ImageUtils.normalize(grayscale_image), (96, 96)))
    face_features = RecogitionService.extract_features(normalized_image)

    return jsonify({"model": list(face_features)})

@app.route('/recognize', methods=['POST'])
def handle_recognize_face_from_features():
    features = np.array(request.json['model'])
    face_to_recognize = ImageUtils.decode_image_buffer(bytearray(request.json['face']['data']))
    grayscale_image = ImageUtils.rgb_to_grayscale(face_to_recognize)
    normalized_image = ImageUtils.grayscale_to_rgb(cv2.resize(ImageUtils.normalize(grayscale_image), (96, 96)))
    face_to_recognize_features = RecogitionService.extract_features(normalized_image)
    are_same = RecogitionService.recognize(features, face_to_recognize_features)

    return jsonify({"areSame": are_same})

app.run(port=4003)
