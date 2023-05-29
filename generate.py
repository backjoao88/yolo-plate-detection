from zipfile import ZipFile
import os
import re
import pandas as pd
import argparse
import itertools
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import matplotlib.pyplot as plt
from utils import log
import seaborn as sn
import time
import numpy as np

argparse = argparse.ArgumentParser(
    prog='Inference', description='Video Inference with YOLO')

argparse.add_argument('--model', default='yolov3')
argparse.add_argument('--dataset', default='data_ccpd_base')
argparse.add_argument('--datasetInf', default='data_ccpd_challenge')
argparse.add_argument('--uuid', type=str,
                      default='a3e9e62e-a98d-11ed-afa1-0242ac120002')
argparse.add_argument(
    '--drive', type=bool, default=False)
argparse.add_argument(
    '--allDatasets', type=bool, default=False)
argparse.add_argument('--time', type=int, default=0)
argparse.add_argument('--save', type=bool, default=False)

args = argparse.parse_args()

tempFolder = './results/temp'


def authDrive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)


def seekFpsArr(fileContent):
    arrFps = re.findall(
        'Predicted in (\d+.\d+) milli-seconds.', fileContent)
    arrFpsInt = [float(fps) for fps in arrFps]
    return arrFpsInt


def seekFpsSum(fileContent):
    arrFps = re.findall(
        'Predicted in (\d+.\d+) milli-seconds.', fileContent)
    arrFpsInt = [float(fps) for fps in arrFps]
    if (sum(arrFpsInt) == 0):
        return 0
    else:
        return (sum(arrFpsInt) / 1000) / 60  # result in minutes


def seekFps(fileContent):
    arrFps = re.findall(
        'Predicted in (\d+.\d+) milli-seconds.', fileContent)
    arrFpsInt = [float(fps)/1000 for fps in arrFps]
    if (sum(arrFpsInt) == 0):
        return 0
    else:
        return len(arrFpsInt) / sum(arrFpsInt)


def seekMetrics(fileContent):
    arrAp = re.findall('ap = (\d+.\d\d|-nan)', fileContent)
    arrF1 = re.findall('F1-score = (\d+.\d\d|-nan)', fileContent)
    arrTp = re.findall('TP = (\d*)', fileContent)
    arrFn = re.findall('FN = (\d*)', fileContent)
    arrFp = re.findall('FP = (\d*)', fileContent)
    arrRecall = re.findall('recall = (\d+.\d\d|-nan)', fileContent)
    arrIou = re.findall('average IoU = (\d+.\d\d) %', fileContent)
    arrMap50 = re.findall('\(mAP@0.60\) = (\d+.\d+|-nan)', fileContent)

    ap, f1, tp, fn, fp, recall, iou, map50 = 0, 0, 0, 0, 0, 0, 0, 0
    if arrAp.__len__() > 0:
        ap = arrAp[0]
    if arrF1.__len__() > 0:
        f1 = arrF1[0]
    if arrTp.__len__() > 0:
        tp = arrTp[0]
    if arrFn.__len__() > 0:
        fn = arrFn[0]
    if arrFp.__len__() > 0:
        fp = arrFp[0]
    if arrRecall.__len__() > 0:
        recall = arrRecall[0]
    if arrIou.__len__() > 0:
        iou = arrIou[0]
    if arrMap50.__len__() > 0:
        map50 = arrMap50[0]
    return ap, f1, tp, fn, fp, recall, iou, map50
    # return ap, ('{:.5E}'.format(float(f1))), tp, fn, fp, recall, iou, map50


def seekLoss(fileContent, epoch):
    arrLoss = re.findall(
        ' {}: (\d+.\d+|-nan), (-nan|\d+.\d+)'.format(epoch), fileContent)
    return arrLoss[0][1]
    # return ('{:.5E}'.format(float(arrLoss[0][1])))


def prepareFileName(rawFileName):
    return rawFileName.split('/')[-1].replace('yolov3-tiny', 'yolov3tiny').replace('yolov4-tiny', 'yolov4tiny').replace('.weights.txt', '').replace('.cfg.txt', '')


def seekHyperParameters(filename):
    modelName = filename.split('-')[0]

    conv = ''
    abbrModel = ''
    if (modelName == 'yolov3tiny'):
        conv = '13'
        abbrModel = 'YV3T'
    elif (modelName == 'yolov3'):
        conv = '52'
        abbrModel = 'YV3'
    elif (modelName == 'yolov4tiny'):
        conv = '21'
        abbrModel = 'YV4T'

    batch = filename.split('-')[2]
    trainingAlgorithm = filename.split('-')[7]
    learningRate = filename.split('-')[8]
    momentum = filename.split('-')[9]
    width = filename.split('-')[10]
    height = filename.split('-')[11].replace('_3000', '').replace('_10000', '')
    epochs = filename.split('-')[1]

    trimmedLearningRate = learningRate[0:1] + '.' + learningRate[1:]

    return abbrModel, conv, batch, trainingAlgorithm, trimmedLearningRate, momentum, width, height, epochs


def fullPath(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]


def unzip(id, datasetName, datasetInfName, resultsPath, zipPath, regexPattern='.txt'):
    fileName = f'3K-10K-TCC-METRICAS-{datasetName.upper()}-{id}.zip'
    results = []
    with ZipFile(f'{resultsPath}/{fileName}', 'r') as zip:
        listOfFileNames = zip.namelist()
        for fileName in listOfFileNames:
            pattern = re.compile(regexPattern)
            if pattern.match(fileName):
                zip.extract(fileName, zipPath)
                results.append(os.path.abspath(f'{zipPath}/{fileName}'))

    return results


def buildBoxAndWrapper(dataset="weather"):

    bests = []

    if dataset == "weather":
        bests = ["yolov3-10000-64-8000-9000-1-18-adam-001-099-416-416_10000.weights_data_ccpd_weather_fps.txt",
                 "yolov3-tiny-10000-32-8000-9000-1-18-adam-001-0999-416-416_10000.weights_data_ccpd_weather_fps.txt", "yolov4-tiny-10000-32-8000-9000-1-18-adam-001-09-416-416_10000.weights_data_ccpd_weather_fps.txt"]
    else:
        bests = ["yolov3-10000-64-8000-9000-1-18-adam-001-099-416-416_10000.weights_data_ccpd_challenge_fps.txt",
                 "yolov3-tiny-10000-64-8000-9000-1-18-adam-001-0999-416-416_10000.weights_data_ccpd_challenge_fps.txt", "yolov4-tiny-10000-32-8000-9000-1-18-adam-0001-09-416-416_10000.weights_data_ccpd_challenge_fps.txt"]

    f1 = open(bests[0], "r")
    f1content = f1.read()
    f2 = open(bests[1], "r")
    f2content = f2.read()
    f3 = open(bests[2], "r")
    f3content = f3.read()

    data1 = seekFpsArr(f1content)
    data2 = seekFpsArr(f2content)
    data3 = seekFpsArr(f3content)

    models_names = ['yolov3', 'yolov4tiny', 'yolov3tiny']
    models = [data1, data2, data3]

    indexesy3 = []
    values = []
    for i in range(0, len(data1)):
        indexesy3.append('YV3_64_001_099')
    values = models[0]
    s1 = pd.Series(values, index=indexesy3)

    indexesy3t = []
    values = []
    for i in range(0, len(data1)):
        indexesy3t.append('YV3T_64_001_0999 ')
    values = models[1]
    s2 = pd.Series(values, index=indexesy3t)

    indexesy4t = []
    values = []
    for i in range(0, len(data1)):
        indexesy4t.append('YV4T_32_001_09')
    values = models[2]
    s3 = pd.Series(values, index=indexesy4t)

    df1 = s1.to_frame()
    df2 = s2.to_frame()
    df3 = s3.to_frame()

    df = pd.concat([df1, df2, df3])
    df.reset_index(inplace=True)
    df.columns = ['Modelos', 'Tempo (ms)']

    sn.set_style("white")
    sn.set_theme()
    sn.boxplot(x=df['Modelos'], y=df['Tempo (ms)'])

    plt.show()


def buildConfusionMatrix(id, datasetName, modelName, datasetInfName, resultsPath, zipPath):
    os.system('rm -rf results/confusion-matrix')
    os.system(f'mkdir -p results/confusion-matrix')

    filesTesting = unzip(id, datasetName, datasetInfName,
                         os.path.abspath(resultsPath), os.path.abspath(zipPath), r'{}-\d*-(.*)._(3000|10000).weights_{}.txt'.format(modelName, datasetInfName))

    for file in filesTesting:
        df_cm = pd.DataFrame([[0, 0], [0, 0]], index=[
            'Verdadeiro', 'Falso'], columns=['Positivo', 'Negativo'])
        arrTp = []
        arrFn = []
        arrFp = []
        arrTn = []
        fileContent = open(file, 'r').read()
        arrTp = re.findall('TP = (\d*)', fileContent)
        arrFn = re.findall('FN = (\d*)', fileContent)
        arrFp = re.findall('FP = (\d*)', fileContent)
        arrTn = [0]
        log(f'Saved confusion matrix to {file}.')
        plt.clf()
        sn.set(font_scale=1.4)

        tp, fn, fp, tn = 0, 0, 0, 0
        if arrTp.__len__() > 0:
            tp = arrTp[0]
        if arrFn.__len__() > 0:
            fn = arrFn[0]
        if arrFp.__len__() > 0:
            fp = arrFp[0]
        if arrTn.__len__() > 0:
            tn = arrTn[0]

        df_cm = pd.DataFrame([[float(tp), float(tn)], [float(fp), float(fn)]], index=[
            'Verdadeiro', 'Falso'], columns=['Positivo', 'Negativo'])
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 24}, fmt='g')
        plt.savefig(f'results/confusion-matrix/{prepareFileName(file)}.png')


def buildLossGraphs(id, datasetName, modelName, datasetInfName, resultsPath, zipPath):
    filesTraining = unzip(id, datasetName, datasetInfName,
                          resultsPath, zipPath, r'{}-\d*-(.*).cfg.txt'.format(modelName))
    lossesArr = []
    os.system(f'mkdir -p results/graphs')
    fig = plt.figure()
    for file in filesTraining:
        lossesArr = []
        fileContent = open(file, 'r').read()
        values = re.findall(
            ' (\d+): (\d+.\d+)', fileContent)
        for value in values:
            lossesArr.append((float(value[0]), float(value[1])))
        log('Saved loss graph to {}.'.format(
            prepareFileName(file)))
        plt.clf()

        fig, ax = plt.subplots()
        ax.set_ylim([0, 0.5])

        fig.set_size_inches(10, 5)

        plt.plot(*zip(*lossesArr),
                 label=prepareFileName(file))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(
            f'results/graphs/{prepareFileName(file)}-graph-loss.png')
    os.system(f'rm -rf {zipPath}/*')
    return 0


def buildTotalMinutesSum(dataset="weather"):
    bests = []

    if dataset == "weather":
        bests = ["yolov3-10000-64-8000-9000-1-18-adam-001-099-416-416_10000.weights_data_ccpd_weather_fps.txt",
                 "yolov3-tiny-10000-32-8000-9000-1-18-adam-001-0999-416-416_10000.weights_data_ccpd_weather_fps.txt", "yolov4-tiny-10000-32-8000-9000-1-18-adam-001-09-416-416_10000.weights_data_ccpd_weather_fps.txt"]
    else:
        bests = ["yolov3-10000-64-8000-9000-1-18-adam-001-099-416-416_10000.weights_data_ccpd_challenge_fps.txt",
                 "yolov3-tiny-10000-64-8000-9000-1-18-adam-001-0999-416-416_10000.weights_data_ccpd_challenge_fps.txt", "yolov4-tiny-10000-32-8000-9000-1-18-adam-0001-09-416-416_10000.weights_data_ccpd_challenge_fps.txt"]

    f1 = open(bests[0], "r")
    f1content = f1.read()
    f2 = open(bests[1], "r")
    f2content = f2.read()
    f3 = open(bests[2], "r")
    f3content = f3.read()

    data1 = seekFpsSum(f1content)
    data2 = seekFpsSum(f2content)
    data3 = seekFpsSum(f3content)

    y3_time = round(float(data1), 2)
    y3t_time = round(float(data2), 2)
    y4t_time = round(float(data3), 2)
    print(f"> Tempo (min) de inferência para o {dataset.upper()}")
    print(f"{bests[0]}: {y3_time}min")
    print(f"{bests[1]}: {y3t_time}min")
    print(f"{bests[2]}: {y4t_time}min")

    data1 = seekFpsSum(f1content)
    data2 = seekFpsSum(f2content)
    data3 = seekFpsSum(f3content)
    return data1, data2, data3


def buildDataframe(modelName='yolov3', datasetName='data_ccpd_base', datasetInfName='data_ccpd_challenge', rawZipTemp='./results/temp', rawZipFolder='./results/raw', rawZipId='90865d32-9822-4766-93a3-3c9c8136fac6', drive=False):

    os.system(f'rm -rf {rawZipTemp}/*')

    if (drive):
        fileToDownload = f'10K-TCC-METRICAS-{datasetName.upper()}-{rawZipId}.zip'
        gdrive = authDrive()
        fileList = gdrive.ListFile(
            {'q': "'#' in parents and trashed=false"}).GetList()
        modelsFile = None
        for driveFile in fileList:
            try:
                if (driveFile['title'] == fileToDownload):
                    print(f'Downloading {fileToDownload}...')
                    modelsFile = gdrive.CreateFile({'id': driveFile['id']})
                    modelsFile.GetContentFile(
                        f'{rawZipFolder}/{fileToDownload}')
                    print(f'{fileToDownload} downloaded.')
            except Exception as ex:
                print(ex.__name__)
                print('[ERROR] Algum erro ocorreu com a autenticação.')
                return

    try:
        filesTraining = unzip(rawZipId, datasetName, datasetInfName,
                              os.path.abspath(rawZipFolder), os.path.abspath(rawZipTemp), r'{}-\d*-(.*).cfg.txt'.format(modelName))

        filesMetrics = unzip(rawZipId, datasetName, datasetInfName,
                             os.path.abspath(rawZipFolder), os.path.abspath(rawZipTemp), r'{}-\d*-(.*)._(3000|10000).weights_{}.txt'.format(modelName, datasetInfName))

        filesFps = unzip(rawZipId, datasetName, datasetInfName,
                         os.path.abspath(rawZipFolder), os.path.abspath(rawZipTemp), r'{}-\d*-(.*)._(3000|10000).weights_{}_fps.txt'.format(modelName, datasetInfName))

    except Exception as ex:
        print(ex.__name__)
        print('[ERROR] Algum erro ocorreu ao descompactar os arquivos.')
        return

    # df = pd.DataFrame(columns=['Title', 'Base', 'Epochs', 'Conv', 'Batch', 'Training Algorithm', 'Learning Rate',
        #    'Momentum', 'Width', 'Height', 'Loss', 'AP', 'F1', 'TP', 'FN', 'FP', 'Recall', 'IoU', 'mAP@0.60', 'FPS'])

    df = pd.DataFrame(columns=['Title', 'Epochs', 'AP', 'F1', 'FPS'])

    filesTraining = sorted(filesTraining)
    filesMetrics = sorted(filesMetrics)
    filesFps = sorted(filesFps)

    print('=' * 200)
    log('> BUILDING DATAFRAMES...')
    print('=' * 200)
    time.sleep(args.time)
    for (fileNameTraining, fileNameMetrics, fileNameFps) in itertools.zip_longest(filesTraining, filesMetrics, filesFps):
        with open(fileNameTraining, 'r') as file:
            fileTrainingContent = file.read()
        with open(fileNameMetrics, 'r') as file:
            fileMetricsContent = file.read()
        with open(fileNameFps, 'r') as file:
            fileFpsContent = file.read()

        filteredFileName = prepareFileName(fileNameTraining)

        hyperParameters = seekHyperParameters(filteredFileName)
        loss = seekLoss(fileTrainingContent, hyperParameters[8])
        metrics = seekMetrics(fileMetricsContent)
        fps = seekFps(fileFpsContent)

        row = pd.Series(
            {
                'Title': hyperParameters[0] + '_' + hyperParameters[2] + '_' + hyperParameters[4].replace('.', '') + '_' + hyperParameters[5],
                'Epochs': hyperParameters[8],
                'AP': float(metrics[0]),
                'F1': metrics[1],
                'FPS': round(float(fps), 2),
                'Loss': loss
            })

        df = pd.concat([df, row.to_frame().T], ignore_index=True)

    os.system(f'rm -rf {rawZipTemp}/*')
    return df


def buildBestFpsDataframe(df):
    df = df.sort_values(['FPS'], ascending=False)
    return df


def buildBestFpsSerieFromDataframe(df):
    df = buildBestFpsDataframe(df)
    df = df[df.FPS != '-nan']
    return df.iloc[[0], :]


def buildBestApDataframe(df):
    df = df.sort_values(['AP'], ascending=False)
    return df


def buildBestApSerieFromDataframe(df):
    df = buildBestApDataframe(df)
    df = df[df.AP != '-nan']
    return df.iloc[[0], :]


def buildBestLossDataframe(df):
    df = df.sort_values(['Loss'], ascending=True)
    return df


def buildBestLossSerieFromDataframe(df):
    df = buildBestLossDataframe(df)
    df = df[df.Loss != '-nan']
    return df.iloc[[0], :]


buildBoxAndWrapper(dataset="challenge")
# buildTotalMinutesSum(dataset="challenge")

# allModelsDf = buildDataframe(args.model, args.dataset, args.datasetInf, './results/temp', './results/raw',
#                              args.uuid, args.drive)

# if args.allDatasets:
#     datasets = ['data_ccpd_base', 'data_ccpd_challenge', 'data_ccpd_weather']
#     ids = ['940cfc9f-38ad-4131-9c6d-f43cd327d402-001',
#            '6e77ac10-726d-44e6-8648-5c043520587f-003', '90865d32-9822-4766-93a3-3c9c8136fac6-002']
#     for uuid, dataset in zip(ids, datasets):
#         allModelsDf = pd.concat([allModelsDf, buildDataframe(args.model, dataset, './results/temp', './results/raw',
#                                                              uuid, args.drive)], ignore_index=True)

# bestLossesDf = buildBestLossDataframe(allModelsDf)
# bestLossSerie = buildBestLossSerieFromDataframe(bestLossesDf)
# bestApDf = buildBestApDataframe(allModelsDf)
# bestApSerie = buildBestApSerieFromDataframe(bestApDf)
# bestFpsDf = buildBestFpsDataframe(allModelsDf)
# bestFpsSerie = buildBestFpsSerieFromDataframe(bestFpsDf)


# log('> RESULTS FOR INFERENCE DATASET: ' + args.dataset)
# print('=' * 200)

# log('> BUILDING LOSS COMPARISON...')
# print('=' * 200)
# time.sleep(args.time)
# print(bestLossesDf)
# print('=' * 200)

# log('> BUILDING AVERAGE PRECISION COMPARISON...')
# print('=' * 200)
# time.sleep(args.time)
# print(bestApDf)
# print('=' * 200)
# log('> BUILDING FRAME PER SECOND (FPS) COMPARISON...')
# print('=' * 200)
# time.sleep(args.time)
# log(bestFpsDf)
# print('=' * 200)

# if args.save:
#     # os.system(f'rm -rf ./results/dfs')
#     os.system(f'mkdir -p ./results/dfs')
#     bestLossesDf.to_csv(
#         f'./results/dfs/{args.model}_{args.dataset}_losses.csv', index=False)
#     bestLossSerie.to_csv(
#         f'./results/dfs/{args.model}_{args.dataset}_bestLoss.csv', index=False)
#     bestFpsDf.to_csv(
#         f'./results/dfs/{args.model}_{args.dataset}_fps.csv', index=False)
#     bestFpsSerie.to_csv(
#         f'./results/dfs/{args.model}_{args.dataset}_bestFps.csv', index=False)
#     bestApDf.to_csv(
#         f'./results/dfs/{args.model}_{args.dataset}_ap.csv', index=False)
#     bestApSerie.to_csv(
#         f'./results/dfs/{args.model}_{args.dataset}_bestAp.csv', index=False)

# log('> BUILDING LOSS GRAPHS...')
# print('=' * 200)
# time.sleep(args.time)
# buildLossGraphs(args.uuid, args.dataset, args.model, args.datasetInf,
#                 './results/raw', './results/temp')
# print('=' * 200)
# log('> BUILDING CONFUSION MATRIXES...')
# print('=' * 200)
# time.sleep(args.time)
# buildConfusionMatrix(args.uuid, args.dataset, args.model, args.datasetInf,
#                      './results/raw', './results/temp')
# print('=' * 200)
