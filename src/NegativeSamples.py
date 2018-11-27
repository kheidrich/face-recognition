import cv2

def load():
    images = []
    for img in range(1,57):
        images.append(cv2.imread('src/negative-face-samples/' + str(img)  + '.jpg'))
    return images