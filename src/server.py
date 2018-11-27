import os
from flask import Flask, request, jsonify
import ImageUtils
import RecogitionService
import numpy as np
import cv2

app = Flask("recognition-service")

@app.route('/train', methods=['POST'])
def handle_extract_face_features():
    face = ImageUtils.decode_image_buffer(bytearray(request.json['face']['data']))
    aligned_face = RecogitionService.align_face(face);
    face_features = RecogitionService.extract_features(aligned_face)

    return jsonify({"model": list(face_features)})

@app.route('/recognize', methods=['POST'])
def handle_recognize_face_from_features():
    features = np.array(request.json['model'])
    face_to_recognize = ImageUtils.decode_image_buffer(bytearray(request.json['face']['data']))
    aligned_face = RecogitionService.align_face(face_to_recognize);
    face_to_recognize_features = RecogitionService.extract_features(aligned_face)
    are_same = RecogitionService.recognize(features, face_to_recognize_features)

    return jsonify({"areSame": are_same})

app.run(port=4003)
