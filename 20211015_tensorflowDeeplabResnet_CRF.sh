#!/bin/sh

#Correr este script con source drSleepDeeplab.sh (no ejecutar como ./drSleepDeeplab.sh)
#Correr desde la carpeta tensorflow-deeplab-resnet: source /home/vicbr/Dropbox/Scripts/drSleepDeeplab.sh 

#set -e # Any subsequent(*) commands which fail will cause the shell script to exit immediately
RED='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${RED}desactivar si hay algun venv corriendo${NC}"
conda deactivate
echo -e "${RED}crear venv con python${NC}"
yes | conda create -n deeplabDrSleep python=2.7 #Aca se pone solo un =, en pip se ponen dos ==
echo -e "${RED}activar venv${NC}"
conda activate deeplabDrSleep
echo -e "${RED}instalar tensorflow${NC}"
pip install tensorflow==0.12
#Dice TF=1.15 https://stackoverflow.com/questions/59786892/module-tensorflow-has-no-attribute-contrib-for-the-version-tensorflow-2-0
echo -e "${RED}instalar requerimientos${NC}"
pip install -r requirements.txt

#Lo siguiente es para hacer la transformacion
#if false; then
echo -e "${RED}protobuf${NC}"
pip install protobuf==3.16
echo -e "${RED}pydensecrf${NC}"
pip install pydensecrf
echo -e "${RED}correr script train.py${NC}"
python train.py --random-scale
#fi

if false; then
source 20211015_tensorflowDeeplabResnet_CRF.sh
#Descargar el kit de https://www.kaggle.com/lyuxinshuai/vocdevkit
python inference.py dogewarrior.jpg ./deeplab_resnet.ckpt #Algo asi

fi

#basestring is no longer available in Python 3: https://stackoverflow.com/questions/34803467/unexpected-exception-name-basestring-is-not-defined-when-invoking-ansible2

#Esto no lo voy a usar pero por si acaso: cd /home/vicbr/0Magister/TesisDocumentos/Modelos/tensorflow-deeplab-resnet

#Downgrading to protobuf < 3.18 seems to be a good workaround: https://pythonrepo.com/repo/mozilla-services-syncserver-python-miscellaneous

#python convert.py prototxt_and_model/solver.prototxt --caffemodel prototxt_and_model/train_iter20000.caffemodel --data-output-path outputVIC/mynet.py
#python convert.py perro --caffemodel gato --data-output-path caballo


