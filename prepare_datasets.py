from utils import buildObjImagesFolder, copyAllImages, buildObjDataFile, buildObjNamesFile, log

log('Construindo arquivo obj.data CCPD_BASE...')
buildObjDataFile(path=['data_ccpd_base', 'obj.data'])

log('Construindo arquivo obj.data CCPD_CHALLENGE...')
buildObjDataFile(path=['data_ccpd_weather', 'obj.data'])

log('Construindo arquivo obj.data CCPD_WEATHER...')
buildObjDataFile(path=['data_ccpd_challenge', 'obj.data'])

log('Construindo arquivo obj.names para CCPD_BASE...')
buildObjNamesFile(path='data_ccpd_base')

log('Construindo arquivo obj.names para CCPD_CHALLENGE...')
buildObjNamesFile(path='data_ccpd_challenge')

log('Construindo arquivo obj.names para CCPD_WEATHER...')
buildObjNamesFile(path='data_ccpd_weather')

log('Construindo arquivo train.txt para CCPD_BASE')
buildObjImagesFolder(
    type='train', pathToCopy='datasets/ccpd_base', pathToWrite='data_ccpd_base')

log('Construindo arquivo valid.txt para CCPD_BASE')
buildObjImagesFolder(type='valid', pathToCopy='datasets/ccpd_base',
                     pathToWrite='data_ccpd_base')

log('Construindo arquivo valid.txt para CCPD_CHALLENGE')
buildObjImagesFolder(type='valid', pathToCopy='datasets/ccpd_challenge',
                     pathToWrite='data_ccpd_challenge')

log('Construindo arquivo valid.txt para CCPD_WEATHER')
buildObjImagesFolder(
    type='valid', pathToCopy='datasets/ccpd_weather', pathToWrite='data_ccpd_weather')


log('Copiando imagens para CCPD_BASE')
copyAllImages(path='data_ccpd_base',
              datasetFolder='ccpd_base')

log('Copiando imagens para CCPD_CHALLENGE')
copyAllImages(path='data_ccpd_challenge',
              datasetFolder='ccpd_challenge')

log('Copiando imagens para CCPD_WEATHER')
copyAllImages(path='data_ccpd_weather',
              datasetFolder='ccpd_weather')
