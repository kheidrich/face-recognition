import os
from openface.openface import AlignDlib, TorchNeuralNet, data

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
face_landmark_model_file = os.path.join(modelDir, 'shape_predictor_68_face_landmarks.dat')

face_aligner = AlignDlib(face_landmark_model_file)

def align_face(rgb_image):
    return face_aligner.align(96, rgb_image)