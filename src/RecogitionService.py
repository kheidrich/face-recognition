import os
import json
from openface import AlignDlib, TorchNeuralNet, data

with open('./src/config.json') as file:
    config = json.load(file)

face_aligner = AlignDlib(config['faceLandmarksModelPath'])
feature_extractor = TorchNeuralNet(config['featureExtractionModelPath'])

def align_face(image):
    return face_aligner.align(96, image)

def extract_features(image):
    return feature_extractor.forward(image)