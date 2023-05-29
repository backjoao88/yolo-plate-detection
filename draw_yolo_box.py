import numpy as np
import os
import cv2

img_path = os.path.join('data_ccpd_weather', 'obj',
                        '0185-3_2-287&611_483&690-483&679_288&690_287&622_482&611-0_0_24_29_25_30_33-57-20.jpg')
txt_path = os.path.join('data_ccpd_weather', 'obj',
                        '0185-3_2-287&611_483&690-483&679_288&690_287&622_482&611-0_0_24_29_25_30_33-57-20.txt')


def getbbox(img, txt):
    lines = txt.readline().splitlines()
    line = lines[0].split()

    height, width, shape = img.shape

    x_center, y_center, w, h = (float(
        line[1])*width, float(line[2])*height, float(line[3])*width, float(line[4])*height)
    x1 = round(x_center-w/2)
    y1 = round(y_center-h/2)
    x2 = round(x_center+w/2)
    y2 = round(y_center+h/2)

    x = (x1, y1, x2, y2)

    return (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))


def drawbbox(img, bbox, output_path):
    cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255, 255), 1)
    cv2.imwrite(output_path, img)


img = cv2.imread(img_path)
txt = open(txt_path)

bbox = getbbox(img, txt)
print(bbox)

drawbbox(img, bbox, os.path.join('outputs', 'images', 'output.jpg'))
