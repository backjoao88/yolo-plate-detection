#!/bin/bash

# Script Auxiliar

TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`

echo "$timestamp INFO     Criando $1 pastas/arquivos..."

# mkdir -p backup
mkdir -p $1
mkdir -p $1/obj
touch $1/obj.data
touch $1/train.txt
touch $1/valid.txt

echo "$timestamp INFO     Copiando imagens do dataset..."

cp `pwd`/datasets/$2/train/_darknet.labels $1/obj.names 2>/dev/null

cp `pwd`/datasets/$2/train/*.jpg $1/obj/ 2>/dev/null
cp `pwd`/datasets/$2/valid/*.jpg $1/obj/ 2>/dev/null
cp `pwd`/datasets/$2/test/*.jpg $1/obj/ 2>/dev/null

echo "$timestamp INFO     Copiando arquivos de txt..."
cp `pwd`/datasets/$2/train/*.txt $1/obj/ 2>/dev/null
cp `pwd`/datasets/$2/valid/*.txt $1/obj/ 2>/dev/null
cp `pwd`/datasets/$2/test/*.txt $1/obj/ 2>/dev/null

echo "$timestamp INFO     Script finalizado com sucesso!"