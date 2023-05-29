# Author: João Paulo Back @ 2022
# python3 inference_video.py --video=tests/videos/0001.mp4 --cfg=tests/cfgs/yolov3-tiny-2000-64-1600-1800-1-18-adam-00001-099-224-224.cfg --weights=tests/weights/yolov3-tiny-2000-64-1600-1800-1-18-adam-00001-099-224-224_2000.weights

import cv2
import argparse
import numpy as np
import os
from utils import predictAndApplyNMS, drawImage, drawImageFps

argparse = argparse.ArgumentParser(
    prog='Inference', description='Video Inference with YOLO')

argparse.add_argument('--video', default='./tests/videos/0001.mp4')
argparse.add_argument('--cfg', default='./tests/cfgs/0001.cfg')
argparse.add_argument('--weights', default='./tests/weights/0001.weights')

args = argparse.parse_args()

videoCapture = cv2.VideoCapture(args.video)
videoCaptureShape = (int(videoCapture.get(
    cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

total = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))

print('[INFO] Reading video {} from disk.'.format(args.video))
videoWriter = cv2.VideoWriter(
    os.path.join('outputs', 'videos', 'output.avi'), cv2.VideoWriter_fourcc(*'XVID'), 15.0, (videoCaptureShape[0], videoCaptureShape[1]))

print('[INFO] Loading YOLO {} from disk.'.format(args.cfg))
cnn = cv2.dnn.readNetFromDarknet(args.cfg, args.weights)
elapsedTime = 0
totalFps = 0

while True:
    hasFrame, frame = videoCapture.read()
    if not frame is None:
        boxes, confidences, _, idxs, time, fps = predictAndApplyNMS(frame, cnn)
    elapsedTime = elapsedTime + time
    totalFps = totalFps + fps
    outputFrame = drawImageFps(frame, fps)
    outputFrame = drawImage(frame, boxes, confidences, idxs)
    if hasFrame:
        videoWriter.write(outputFrame)
    else:
        break

print('[INFO] All frames took {:.4f} seconds to end (Avg FPS={:.2f})'.format(
    elapsedTime, (totalFps/total)))

videoCapture.release()
cv2.destroyAllWindows()
