import cv2

def load():
    images = []
    for img in range(1,15):
        images.append(cv2.imread('src/negative-face-samples/subject' + str(img)  + '.jpg'))
    return images