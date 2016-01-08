import os,sys, subprocess
sys.path.insert(0, os.getcwd()+'/python/my_script/')

from tester import tester
from trainer import trainer
from crf_runner import crf_runner, grid_search
import tools

# MODIFY PATH for YOUR SETTING
EXP='voc12' #dataset
NET_ID='vgg128_noup' #model name
NUM_LABELS=21
YEAR = 'VOC2012'
DATA_ROOT='/media/ali/VOCdevkit/VOC2012'
#DATA_ROOT=subprocess.Popen('cd .. && pwd', stdout=subprocess.PIPE, shell=True).communicate()[0][:-1] + '/VOCdevkit/' + YEAR
OLD_ROOT=''#only change if you are changing the path to images
DEV_ID=0 #gpu id
LOAD_MAT_FILE = 1
       
train_set_SUFFIX='_aug'

train_set_STRONG='train'
#train_set_STRONG='train200'
#train_set_STRONG='train500'
#train_set_STRONG='train1000'
#train_set_STRONG='train750'

train_set_WEAK_LEN=0 #'5000'


# Run

RUN_TRAIN=1 # Training #1 (on train_aug)
RUN_TEST=0  # Test #1 specification (on val or test)
RUN_TRAIN2=0 # Training #2 (finetune on trainval_aug)
RUN_TEST2=0 # Test #2 on official test set
RUN_SAVE=0 # Translate and save the model
RUN_DENSECRF=0 # To Run Densecrf
GRID_SEARCH=0 # To Run ONLY if you dont know what parameters to use for Densecrf




#####

def env_creater():
    dic = {'EXP': EXP, 'NET_ID': NET_ID, 'NUM_LABELS': NUM_LABELS, 'DATA_ROOT': DATA_ROOT, 'DEV_ID':DEV_ID, 'OLD_ROOT': OLD_ROOT}
    dic.update({'train_set_SUFFIX': train_set_SUFFIX, 'train_set_STRONG': train_set_STRONG, 'train_set_WEAK_LEN': train_set_WEAK_LEN})
    dic.update({'year': YEAR, 'POSTPROCESS': 0})
    tools.environment_variable_creator(dic)


def run(RUN_TRAIN, RUN_TEST, RUN_TRAIN2, RUN_TEST2, RUN_SAVE):
    tools.mkdir()
    if RUN_TRAIN : trainer()
    if RUN_TEST : tester()
    if RUN_TRAIN2 : trainer(type_=2)
    if RUN_TEST2 : tester(type_=2)
    if RUN_SAVE: tools.saver()
    if RUN_DENSECRF : crf_runner(LOAD_MAT_FILE, RUN_TRAIN2)
    if GRID_SEARCH : grid_search(LOAD_MAT_FILE, RUN_TRAIN2) 


if __name__ == "__main__":
    env_creater()

    run(RUN_TRAIN, RUN_TEST, RUN_TRAIN2, RUN_TEST2, RUN_SAVE)
