from flask import Flask, request, jsonify

app = Flask("recognition-service")

@app.route('/train', methods=['POST'])
def handle_train_model():
    return jsonify({"model": {"data": [22,22,22,22], "type": "Buffer"}})

@app.route('/compare', methods=['POST'])
def handle_compare_image_with_model():
    return jsonify({"areSame": True})

app.run(port=4003)
