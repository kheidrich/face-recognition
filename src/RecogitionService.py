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
    bb = face_aligner.getLargestFaceBoundingBox(image)
    return face_aligner.align(96, image, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


def extract_features(image):
    return feature_extractor.forward(image)

def recognize(face_features, face_features_to_recognize):
    NEGATIVE = 0
    POSITIVE = 1
    train_data = list(negative_samples)
    labels = []
    knn = cv2.ml.KNearest_create()
    for i in range(0, len(negative_samples)):
        labels.append(NEGATIVE)
    train_data.append(np.array(face_features, dtype='f'))
    labels.append(POSITIVE)
    knn.train(np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(labels))
    (ret, results, neighbours, dist) = knn.findNearest(np.array([np.array(face_features_to_recognize, dtype='f')]), 2)
    distance = dist[0][0]
    most_nearest = neighbours[0][0]
    if(most_nearest == 1 and distance <= config['distanceThreshold']):
        return True
    else:
        return False 

negative_samples = []
samples = NegativeSamples.load()
for sample in samples:
        features = np.array(extract_features(align_face(sample)))
        negative_samples.append(np.array(features, dtype='f'))
samples = None