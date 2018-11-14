import os
import json
import cv2
from openface import AlignDlib, TorchNeuralNet, data
import NegativeSamples
import numpy as np

with open('./src/config.json') as file:
    config = json.load(file)

face_aligner = AlignDlib(config['faceLandmarksModelPath'])
feature_extractor = TorchNeuralNet(config['featureExtractionModelPath'])

def align_face(image):
    return face_aligner.align(96, image)


def extract_features(image):
    return feature_extractor.forward(image)

def recognize(face_features, face_features_to_recognize):
    NEGATIVE = 0
    POSITIVE = 1
    samples = NegativeSamples.load()
    train_data = []
    labels = []
    knn = cv2.ml.KNearest_create()
    for sample in samples:
        features = np.array(extract_features(align_face(sample)))
        train_data.append(np.array(features, dtype='f'))
        labels.append(NEGATIVE)
    train_data.append(np.array(face_features, dtype='f'))
    labels.append(POSITIVE)
    knn.train(np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(labels))
    print knn.findNearest(np.array([np.array(face_features_to_recognize, dtype='f')]), 3)
    return False
