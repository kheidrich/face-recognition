import os
from flask import Flask, request, jsonify
import ImageUtils
import RecogitionService
import cv2

app = Flask("recognition-service")

@app.route('/train', methods=['POST'])
def handle_train_model():
    image = ImageUtils.to_rgb_array(bytearray(request.json['face']['data']))
    grayscale_image = ImageUtils.rgb_to_grayscale(image)
    normalized_image = ImageUtils.normalize(grayscale_image)
    aligned_face = ImageUtils.grayscale_to_rgb(RecogitionService.align_face(normalized_image))

    tmp_image = open('/mnt/c/teste/tmp.jpg', 'wb')
    (done, encoded) = cv2.imencode('.jpg', aligned_face)
    tmp_image.write(encoded)
    tmp_image.close()

    print(RecogitionService.extract_features(aligned_face))

    return jsonify({"model": {"data": [22,22,22,22], "type": "Buffer"}})

@app.route('/recognize', methods=['POST'])
def handle_compare_image_with_model():
    return jsonify({"areSame": True})

app.run(port=4003)
