import os
from openface.openface import data

fileDir = os.path.dirname(os.path.realpath(__file__))

def to_rgb_array(image_buffer):
    tmp_image = open('image.bin', 'wb')
    tmp_image.write(image_buffer)
    print(os.path.join(fileDir, '..', 'image.bin'))
    rgb = data.Image('tmp', 'imagetmp', os.path.join(fileDir, '..', 'image.bin')).getRGB()
    tmp_image.close()
    os.remove('image.bin')
    return rgb