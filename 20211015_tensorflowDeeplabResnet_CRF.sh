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
#Descargar el kit desde: https://www.kaggle.com/lyuxinshuai/vocdevkit
python inference.py personas.jpg ./deeplab_resnet.ckpt #Algo asi para correr CRF
#Bajar checkpoints desde: https://drive.google.com/drive/folders/0B_rootXHuswsZ0E4Mjh1ZU5xZVU?resourcekey=0-9Ui2e1br1d6jymsI6UdGUQ
fi

#basestring is no longer available in Python 3: https://stackoverflow.com/questions/34803467/unexpected-exception-name-basestring-is-not-defined-when-invoking-ansible2
#Downgrading to protobuf < 3.18 seems to be a good workaround: https://pythonrepo.com/repo/mozilla-services-syncserver-python-miscellaneous

