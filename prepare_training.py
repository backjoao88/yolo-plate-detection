from utils import buildDarknetCfgFiles, log
import argparse

argparse = argparse.ArgumentParser(
    prog='Inference', description='Test Dataset')

argparse.add_argument('--model', type=str, default='yolov3-tiny')

args = argparse.parse_args()

log('Criando arquivos de configuração de treinamento {}...'.format(args.model))
buildDarknetCfgFiles(args.model)
