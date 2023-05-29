import os
import os
import glob
import logging
import argparse

argparse = argparse.ArgumentParser(
    prog='Inference', description='Test Dataset')

argparse.add_argument('--data', type=str, default='data_ccpd_base')
argparse.add_argument('--on', type=str, default='data_ccpd_challenge')
argparse.add_argument('--iou', type=str, default='0.5')

args = argparse.parse_args()

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def log(message):
    logging.info(message)


def test(datasetDirInference, datasetName, modelsDir, datasetInferenceName, iou=0.5):
    models = glob.glob(modelsDir)
    os.system('chmod +777 ./test.sh')
    os.system('chmod +777 ./test_fps.sh')
    log('Modelos da base {} sendo testados em {} (IoU = {})'.format(
        datasetName, datasetDirInference, iou))

    for model in models:
        cfgDir = model.replace(".weights", ".cfg").replace(
            "/{}/backup".format(datasetName), "/models").replace("_3000", "").replace("_10000", "")
        log('Testando o modelo do dataset {}: {}, cfg: {}'.format(
            datasetDirInference, model, cfgDir))
        # os.system('./test.sh {} {} {} {} {}'.format(model,
        #           datasetDirInference, cfgDir, datasetInferenceName, iou))
        os.system('./test_fps.sh {} {} {} {}'.format(model,
                  datasetDirInference, cfgDir, datasetInferenceName))


ABS_PATH = os.getcwd()

datasetName = args.data
datasetInference = args.on
iou = args.iou

datasetDirInference = os.path.join(ABS_PATH, '..', datasetInference)

modelsDir = os.path.join(
    ABS_PATH, '..', datasetName, 'backup', '*.weights')

test(datasetDirInference, datasetName, modelsDir, datasetInference, iou)
