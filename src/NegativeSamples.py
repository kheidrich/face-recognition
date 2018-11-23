import cv2

images = []
for img in range(0,39):
    images.append(cv2.imread('src/negative-face-samples/1~' + str(img)  + '.jpg'))

for img in range(1,15):
    images.append(cv2.imread('src/negative-face-samples/1~' + str(img)  + '.jpg'))

def load():
    return images