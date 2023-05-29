# @author João Paulo Back

import os
import logging
import cv2
import requests
import re
import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------------------------------------------------
# Logging Auxiliary Functions
# ----------------------------------------------------------------------

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def log(message):
    logging.info(message)

# ----------------------------------------------------------------------
# CCPD Datasets Auxiliary Functions
# ----------------------------------------------------------------------


def getAllCCPDBoundingBoxes(filenames):
    boudingBoxes = filenames.split("-")[3].split("_")

    rightBottomX = int(boudingBoxes[0].split("&")[0])
    rightBottomY = int(boudingBoxes[0].split("&")[1])

    leftBottomX = int(boudingBoxes[1].split("&")[0])
    leftBottomY = int(boudingBoxes[1].split("&")[1])

    leftUpperX = int(boudingBoxes[2].split("&")[0])
    leftUpperY = int(boudingBoxes[2].split("&")[1])

    rightUpperX = int(boudingBoxes[3].split("&")[0])
    rightUpperY = int(boudingBoxes[3].split("&")[1])

    return min(rightBottomX, leftBottomX, leftUpperX, rightUpperX), min(rightBottomY, leftBottomY, leftUpperY, rightUpperY), max(rightBottomX, leftBottomX, leftUpperX, rightUpperX), max(rightBottomY, leftBottomY, leftUpperY, rightUpperY)


def generateCCPDYoloAnnotations(datasetFolder):
    datasetFilenames = os.listdir(datasetFolder)

    for filename in datasetFilenames:
        img = cv2.imread(os.path.join(datasetFolder, filename))
        height = img.shape[0]
        width = img.shape[1]

        boundingBox = getAllCCPDBoundingBoxes(filename)
        leftUpper = (boundingBox[0], boundingBox[1])
        rightBottom = (boundingBox[2], boundingBox[3])

        x = (rightBottom[0] + leftUpper[0]) / 2 / width
        y = (rightBottom[1] + leftUpper[1]) / 2 / height
        w = (rightBottom[0] - leftUpper[0]) / width
        h = (rightBottom[1] - leftUpper[1]) / height

        stringFormat = "0 " + str(x) + " " + str(y) + \
            " " + str(w) + " " + str(h)
        stringFile = open(os.path.join(
            datasetFolder, filename.split(".")[0] + ".txt"), 'a')
        stringFile.write(stringFormat)
        stringFile.close()

    stringFormat = "0"
    stringFile = open(os.path.join(datasetFolder, "_darknet.labels"), 'a')
    stringFile.write(stringFormat)
    stringFile.close()


def generateAllCCPDYoloAnnotations(trainingDatasets=[], testingDatasets=[]):
    if trainingDatasets != []:
        for dataset in trainingDatasets:
            generateCCPDYoloAnnotations(f'datasets/{dataset}/train')

    if testingDatasets != []:
        for dataset in testingDatasets:
            generateCCPDYoloAnnotations(f'datasets/{dataset}/test')

    if testingDatasets != []:
        for dataset in testingDatasets:
            generateCCPDYoloAnnotations(f'datasets/{dataset}/valid')


def drawCCPDBoundingBox(imgName):
    img = cv2.imread(os.path.join(imgName))
    boundingBox = getAllCCPDBoundingBoxes(imgName)
    leftUpper = (boundingBox[0], boundingBox[1])
    rightBottom = (boundingBox[2], boundingBox[3])
    cv2.rectangle(img, (leftUpper[0], leftUpper[1]),
                  (rightBottom[0], rightBottom[1]), (0, 0, 255), 1)
    cv2.imshow(imgName, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

# ----------------------------------------------------------------------
# Darknet Yolo Annotation Format Auxiliary Functions
# ----------------------------------------------------------------------


def buildObjDataFile(config='', path=['data_lprbr_base', 'obj.data']):
    pathToWrite = os.path.join(path[0], path[1])
    ABS_PATH = os.getcwd()
    if config == '':
        train_path = os.path.join(ABS_PATH, path[0], 'train.txt')
        valid_path = os.path.join(ABS_PATH, path[0], 'valid.txt')
        obj_names_path = os.path.join(ABS_PATH, path[0], 'obj.names')
        backup_path = os.path.join(ABS_PATH, path[0], 'backup')
        classes = 1
        config = {
            "classes": classes,
            "train": train_path,
            "valid": valid_path,
            "names": obj_names_path,
            "backup": backup_path,
        }

    if "classes" and "train" and "valid" and "names" and "backup" in config:
        classes = config["classes"]
        train_path = config["train"]
        valid_path = config["valid"]
        obj_names_path = config["names"]
        backup_path = config["backup"]

    json_cfg_obg_txt = "classes = {}\ntrain = {}\nvalid = {}\nnames = {}\nbackup = {}\n".format(
        classes, train_path, valid_path, obj_names_path, backup_path)
    with open(pathToWrite, 'w') as out:
        out.write(json_cfg_obg_txt)


def buildObjNamesFile(names=["license_plate"], path='data_lprbr_base'):
    with open(os.path.join(path, 'obj.names'), 'w') as out:
        out.write(names[0])


def buildObjImagesFolder(type='train', pathToCopy='datasets/lprbr_base', pathToWrite='data_lprbr_base'):
    ABS_PATH = os.getcwd()
    with open('{}/{}.txt'.format(pathToWrite, type), 'w') as out:
        for img in [f for f in os.listdir('{}/{}/{}/'.format(os.getcwd(), pathToCopy, type)) if f.endswith('jpg')]:
            out.write('{}/{}/obj/'.format(ABS_PATH, pathToWrite) + img + '\n')


def copyAllImages(path='data_lprbr_base', datasetFolder='lprbr_base'):
    ABS_PATH = os.getcwd()
    os.system('chmod +777 {}/scripts/copyImgs.sh'.format(ABS_PATH))
    os.system('{}/scripts/copyImgs.sh {} {}'.format(ABS_PATH, path, datasetFolder))

# ----------------------------------------------------------------------
# Training Auxiliary Functions
# ----------------------------------------------------------------------


def imShow(path):
    image = cv2.imread(path)
    height, width = image.shape[:2]
    resized_image = cv2.resize(
        image, (3*width, 3*height), interpolation=cv2.INTER_CUBIC)

    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.show()


def buildDarknetCfgFiles(type='yolov3-tiny'):

    parametersSearchSpace = {
        'optimizer': ['adam'],
        'batch': [32, 64],
        'learning_rate': [0.01, 0.001],
        'momentum': [0.9, 0.99, 0.999],
        'resolution': [(416, 416)]
    }

    for o in parametersSearchSpace['optimizer']:
        for b in parametersSearchSpace['batch']:
            for lr in parametersSearchSpace['learning_rate']:
                for m in parametersSearchSpace['momentum']:
                    for w, h in parametersSearchSpace['resolution']:
                        max_batch = 10000
                        if type == 'yolov3':
                            max_batch = 10000
                        step1 = 0.8 * max_batch
                        step2 = 0.9 * max_batch

                        config = {
                            'batch': b,
                            'max_batch': max_batch,
                            'step1': 0.8 * max_batch,
                            'step2': 0.9 * max_batch,
                            'num_classes': 1,
                            'num_filters': (1 + 5) * 3,
                            'optimizer': o,
                            'learning_rate': lr,
                            'momentum': m,
                            'resolution': (w, h)
                        }
                        buildDarknetCfgFile(type, config)
    return True


def buildDarknetCfgFile(type='yolov3-tiny', config='', path=['models']):
    pathToWrite = os.path.join(path[0])
    if config == '':
        config = {
            'batch': 64,
            'max_batch': 2000,
            'step1': 0.8 * 2000,
            'step2': 0.9 * 2000,
            'num_classes': 1,
            'num_filters': (1 + 5) * 3,
            'optimizer': 'sgd',
            'learning_rate': 0.001,
            'momentum': 0.9,
            'resolution': (416, 416)
        }

    valid_models = ['yolov3', 'yolov4', 'yolov3-tiny', 'yolov4-tiny']
    if type not in valid_models:
        raise Exception('[ERROR] Type a valid model to download from Darknet.')

    cfg_new_filename = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(type, config['max_batch'], int(config['batch']), int(config['step1']), int(
        config['step2']), config['num_classes'], config['num_filters'], config['optimizer'], "{}".format(config['learning_rate']).replace(".", ""), str(config['momentum']).replace(".", ""), config['resolution'][0], config['resolution'][1])
    cfg_path_to_save = '{}/{}/{}.cfg'.format(
        os.getcwd(), pathToWrite, cfg_new_filename)
    cfg_base_url = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/{}.cfg'.format(
        type)
    cfg_content = requests.get(cfg_base_url).text

    cfg_content = re.sub('max_batches = \d*', 'max_batches = ' +
                         str(config['max_batch']), cfg_content)
    cfg_content = re.sub('batch=\d*', 'batch = ' +
                         str(config['batch']), cfg_content)
    cfg_content = re.sub('subdivisions=\d*', 'subdivisions=2', cfg_content)
    cfg_content = re.sub('steps=\d*,\d*', 'steps='+"{:.0f}".format(
        config['step1'])+','+"{:.0f}".format(config['step2']), cfg_content)
    cfg_content = re.sub('classes=\d*', 'classes=' +
                         str(config['num_classes']), cfg_content)
    cfg_content = re.sub('pad=1\nfilters=\d*', 'pad=1\nfilters=' +
                         "{:.0f}".format(config['num_filters']), cfg_content)
    cfg_content = re.sub('learning_rate=\d*.\d*', 'learning_rate=' +
                         str(config['learning_rate']), cfg_content)
    cfg_content = re.sub('momentum=\d*.\d*', 'momentum=' +
                         str(config['momentum']), cfg_content)
    cfg_content = re.sub('width=\d*', 'width=' +
                         str(config['resolution'][0]), cfg_content)
    cfg_content = re.sub('height=\d*', 'height=' +
                         str(config['resolution'][1]), cfg_content)

    if (config['optimizer'] == 'adam'):
        cfg_content = re.sub('hue=.1\n', 'hue=.1\nadam=1\n', cfg_content)

    os.makedirs('{}/{}'.format(os.getcwd(), pathToWrite), exist_ok=True)

    log('Construindo arquivo {}...'.format(cfg_path_to_save))
    with open(cfg_path_to_save, 'w') as cfg_file:
        cfg_file.write(cfg_content)

    return cfg_content

# -----------------------------------------------------------------------


def getOutputLayers(net):
    """Function that return the last layers of the parameterized net"""
    layerNames = net.getLayerNames()
    outputLayers = [layerNames[i - 1]
                    for i in net.getUnconnectedOutLayers()]
    return outputLayers


def predictAndApplyNMS(img, model, size=(416, 416), confidenceThresh=0.6, nmsThresh=0.6):

    blob_image = cv2.dnn.blobFromImage(
        img, 1/255, size, (0, 0, 0), True, crop=False)
    model.setInput(blob_image)
    predictedOutputs, start, end = predict(model)

    print('[INFO] Frame prediction took {:.4f} seconds to end (FPS={:.2f})'.format(
        end-start, 1/(end-start)))
    outputs = handlePredictions(
        predictedOutputs, imgSize=img.shape, confidenceThresh=confidenceThresh)
    idxs = cv2.dnn.NMSBoxes(
        outputs[0], outputs[1], confidenceThresh, nmsThresh)
    return outputs[0], outputs[1], outputs[2], idxs, (end-start), (1/(end-start))


def predict(model):
    start = time.time()
    outputs = model.forward(getOutputLayers(model))
    end = time.time()
    return [outputs, start, end]


def handlePredictions(outputs, imgSize, confidenceThresh=0.5):
    boxes = []
    confidences = []
    classIds = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidenceThresh:
                box_scaled = detection[0:4] * np.array(
                    [imgSize[1], imgSize[0], imgSize[1], imgSize[0]])
                (centerX, centerY, w, h) = box_scaled.astype('int')
                x = int(centerX - (w/2))
                y = int(centerY - (h/2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                classIds.append(classId)
    return [boxes, confidences, classIds]

def drawImgTitle(image, modelName="YV4T_32_001_09"):

    x,y = image.shape[1], image.shape[0]

    #Title Rec
    title_rec_location = (0,0)
    title_rec_size = (x+y+200, 50)
    title_rec_color= (255,255,255)

    #Title
    title_text = modelName
    title_font = cv2.FONT_HERSHEY_SIMPLEX
    title_fontScale = 1
    title_fontColor = (0,0,0)
    title_lineType = 2
    
    text_width, text_height = cv2.getTextSize(title_text, title_font, title_fontScale, title_lineType)[0]
    title_location = (int(image.shape[1] / 2)-int(text_width / 2), int((title_rec_size[1]+text_height+20) / 2) - int(text_height / 2)) 

    cv2.rectangle(image,title_rec_location,title_rec_size,title_rec_color, -1)
    cv2.putText(image, title_text, title_location, title_font, title_fontScale, title_fontColor, title_lineType)

    return image


def drawImage(image, boxes, confidences, idxs):

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)


            #Conf Text
            cf_text = "Conf: {:.4f}".format(round(confidences[i], 3))
            cf_font = cv2.FONT_HERSHEY_SIMPLEX
            cf_fontScale = 2
            cf_fontColor = (0,0,255)
            cf_lineType = 3
            cf_text_width, cf_text_height = cv2.getTextSize(cf_text, cf_font, cf_fontScale, cf_lineType)[0]
            cf_text_location = (int(image.shape[1] / 2)-int(cf_text_width / 2), int((image.shape[1]+cf_text_height+20) / 2) + y+40)

            #Label Text
            l_text = "{}".format("LICENSE_PLATE") 
            l_font = cv2.FONT_HERSHEY_SIMPLEX
            l_fontScale = 1
            l_fontColor = (0,0,255)
            l_lineType = 3
            l_text_location = (x, y-5)
    

            cv2.putText(image, cf_text, cf_text_location, cf_font, cf_fontScale, cf_fontColor, cf_lineType)
            cv2.putText(image, l_text, l_text_location,l_font, l_fontScale, l_fontColor, l_lineType)

    return image


def drawImageFps(image, currentFps=0):
    fpsText = "FPS: {:.2f}".format(currentFps)
    cv2.putText(image, fpsText, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return image
