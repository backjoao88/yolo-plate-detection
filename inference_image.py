# Author: João Paulo Back @ 2022
# python3 inference_image.py --img=tests/images/0001.jpg --cfg=tests/cfgs/yolov3-tiny-2000-64-1600-1800-1-18-adam-00001-099-224-224.cfg --weights=tests/weights/yolov3-tiny-2000-64-1600-1800-1-18-adam-00001-099-224-224_2000.weights

import cv2
import argparse
import numpy as np
from utils import predictAndApplyNMS, drawImage, drawImgTitle
import os

argparse = argparse.ArgumentParser(
    prog='Inference', description='Image Inference with YOLO')

argparse.add_argument('--img', default='./tests/images/0001.jpg')
argparse.add_argument('--cfg', default='./tests/cfgs/0001.cfg')
argparse.add_argument('--weights', default='./tests/weights/0001.weights')
argparse.add_argument('--modelName', default='YV3_64_001_099')
argparse.add_argument('--outputFilename', default='output_6_yolov3t_first.jpg')

args = argparse.parse_args()


print('[INFO] Reading image {} from disk.'.format(args.img))
img = cv2.imread(args.img)

print('[INFO] Loading YOLO {} from disk.'.format(args.cfg))
cnn = cv2.dnn.readNetFromDarknet(args.cfg, args.weights)

elapsedTime = 0
totalFps = 0

boxes, confidences, _, idxs, time, fps = predictAndApplyNMS(img, cnn)
elapsedTime = elapsedTime + time
totalFps = totalFps + fps
outputImg = drawImgTitle(img, modelName=args.modelName)
outputImg = drawImage(img, boxes, confidences, idxs)

print('[INFO] All frames took {:.4f} seconds to end (Avg FPS={:.2f})'.format(
    elapsedTime, totalFps))

path = os.path.join('outputs', 'images', args.outputFilename)
print(f"[INFO] Writing to {path}")
cv2.imwrite(path, outputImg)
