import os


import os
import glob
import logging
import argparse

argparse = argparse.ArgumentParser(
    prog='Inference', description='Test Dataset')

argparse.add_argument('--data', type=str, default='data_ccpd_base')

args = argparse.parse_args()

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def log(message):
    logging.info(message)


def train(datasetDir, modelsDir):
    models = glob.glob(modelsDir)
    os.system('chmod +777 ./train.sh')
    print(models)
    for model in models:
        log('Treinando o modelo do dataset {}: {}'.format(datasetDir, model))
        os.system('./train.sh {} {}'.format(model, datasetDir))


# Obs: This training file will be copied to ./darknet folder on Collab Script.
ABS_PATH = os.getcwd()

datasetDir = os.path.join(ABS_PATH, '..', args.data)
modelsDir = os.path.join(ABS_PATH, '..', 'models', '*.cfg')

train(datasetDir, modelsDir)
