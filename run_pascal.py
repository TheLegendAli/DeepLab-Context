import os
import subprocess

# MODIFY PATH for YOUR SETTING
EXP='voc12'
NET_ID='vgg128_noup'
NUM_LABELS=21
DATA_ROOT='../VOCdevkit/VOC2012'

DEV_ID=0 #gpu id


# Run

RUN_TRAIN=0 # Training #1 (on train_aug)
RUN_TEST=0  # Test #1 specification (on val or test)
RUN_TRAIN2=0 # Training #2 (finetune on trainval_aug)
RUN_TEST2=0 # Test #2 on official test set
RUN_SAVE=0 # Translate and save the model



TRAIN_SET_SUFFIX='_aug'

TRAIN_SET_STRONG='train'
#TRAIN_SET_STRONG='train200'
#TRAIN_SET_STRONG='train500'
#TRAIN_SET_STRONG='train1000'
#TRAIN_SET_STRONG='train750'

TRAIN_SET_WEAK_LEN=0 #'5000'



# Create dirs
CAFFE_DIR='./'
CAFFE_BIN='.build_release/tools/caffe.bin'

CONFIG_DIR=EXP + '/config/' + NET_ID
MODEL_DIR=EXP + '/model/' + NET_ID
LOG_DIR=EXP  + '/log/' + NET_ID
LIST_DIR=EXP + '/list'
if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

os.environ['GLOG_log_dir'] = LOG_DIR


def model_finder(path, type_=1):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    files= list(sorted(os.listdir(path), key=mtime))
    for fil in files:
        if type_==1:
            if fil[0:10] == 'train_iter_' and fil[-11:]=='.caffemodel':
                return fil
        else:
            if fil[0:11] == 'train2_iter_' and fil[-11:]=='.caffemodel':
                return fil

#####

def trainer(type_=1):
    train = 'train' if type_==1 else 'trainval'
    init = '/init.caffemodel' if type_==1 else '/init2.caffemodel'
    solver = 'solver' if type_==1 else 'solver2'

    TRAIN_SET=train + TRAIN_SET_SUFFIX
    if TRAIN_SET_WEAK_LEN ==0:
        TRAIN_SET_WEAK=TRAIN_SET + '_diff_' + TRAIN_SET_STRONG
        file1 = LIST_DIR + "/" + TRAIN_SET + ".txt"
        file2 = LIST_DIR + "/" + TRAIN_SET_STRONG + ".txt"
        file_output = LIST_DIR + "/" + TRAIN_SET_WEAK + ".txt"
        command = 'comm -3 {0} {1} > {2}'.format(file1, file2, file_output)
        subprocess.call(command, shell=True)
    else:
        TRAIN_SET_WEAK= TRAIN_SET + '_diff_' + TRAIN_SET_STRONG + '_head' + TRAIN_SET_WEAK_LEN
        file1 = LIST_DIR + "/" + TRAIN_SET + ".txt"
        file2 = LIST_DIR + "/" + TRAIN_SET_STRONG + ".txt"
        file3 = TRAIN_SET_WEAK_LEN
        file_output = LIST_DIR + "/" + TRAIN_SET_WEAK + ".txt"
        command = 'comm -3 {0} {1} | head -n {2} > {3}'.format(file1, file2, file3, file_output)
        subprocess.call(command, shell=True)

    MODEL=EXP + '/model/' + NET_ID + init #change this
    #if not os.path.isfile(MODEL): MODEL=model_finder(EXP+'/model/'+NET_ID)
    #
    print 'Training' + str(type_) + net + ' ' + EXP + '/' + NET_ID # change this
    for variable in ['train', solver]: #change this
        file1= CONFIG_DIR + '/' + variable + '.prototxt'
        file_output = CONFIG_DIR + '/' + variable + '_' + TRAIN_SET + '.prototxt'
        command = 'sed "$(eval echo $(cat sub.sed))" {0} > {1}'.format(file1, file_output)
        subprocess.call(command, shell=True)
    CMD = CAFFE_DIR + CAFFE_BIN + ' train' \
    ' --solver=' + CONFIG_DIR + '/' + solver + '_' + TRAIN_SET + '.prototxt' \
    ' --weight=' + MODEL + ' --gpu=' + str(DEV_ID) #change solver
    print 'Running ' + CMD
    #subprocess.call(CMD, shell=True)


def tester(type_=1):
    set_ = ['val'] if type_==1 else ['val', 'test']
    caffe_ = '/test.caffemodel' if type_==1 else '/test2.caffemodel'
    features = '/features/' if type_==1 else '/features2/'
    for TEST_SET in set_:
        file1 = LIST_DIR + '/' + TEST_SET + '.txt'
        cmd = 'cat {0} | wc -l'.format(file1)
        TEST_ITER = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0][:-1]

        MODEL=EXP + '/model/' + NET_ID + caffe_
        if not os.path.isfile(MODEL): MODEL=model_finder(EXP+'/model/'+NET_ID, type_)
        
        print 'Testing' + str(type_) + ' net ' + EXP + '/' + NET_ID
        FEATURE_DIR=EXP + features + NET_ID
        fc8 = FEATURE_DIR + '/' + TEST_SET + '/fc8'
        crf = FEATURE_DIR + '/' + TEST_SET + '/crf'
        if not os.path.exists(fc8): os.makedirs(fc8)
        if not os.path.exists(crf): os.makedirs(crf)
        
        file1= CONFIG_DIR + '/test.prototxt'
        file_output = CONFIG_DIR + '/test_' + TEST_SET + '.prototxt'
        command = 'sed "$(eval echo $(cat sub.sed))" {0} > {1}'.format(file1, file_output)
        subprocess.call(command, shell=True)
        
        CMD = CAFFE_DIR + CAFFE_BIN + ' test --model=' + CONFIG_DIR + '/test_' + TEST_SET + '.prototxt' \
        ' --weights=' + MODEL + ' --gpu=' + str(DEV_ID) + ' --iterations=' + str(TEST_ITER)
        print 'Running ' + CMD
        #subprocess.call(CMD, shell=True)

def saver():#doesnt really save
    MODEL=EXP + '/model/' + NET_ID + '/test2.caffemodel'
    #if not os.path.isfile(MODEL): MODEL=model_finder(EXP+'/model/'+NET_ID)
    MODEL_DEPLOY=EXP + '/model/' + NET_ID + '/deploy.caffemodel'

    print 'Translating net ' + EXP + '/' + NET_ID
    CMD = CAFFE_DIR + CAFFE_BIN + ' save --model=' + CONFIG_DIR + '/deploy.prototxt' \
    ' --weights=' + MODEL + ' --out_weight=' + MODEL_DEPLOY
    print 'Running ' + CMD
    #subprocess.call(CMD, shell=True)


def run(RUN_TRAIN, RUN_TEST, RUN_TRAIN2, RUN_TEST2, RUN_SAVE):
    if RUN_TRAIN : trainer()
    if RUN_TEST : tester()
    if RUN_TRAIN2 : trainer(type_=2)
    if RUN_TEST2 : tester(type_=2)
    if RUN_SAVE: saver() 


if __name__ == "__main__":
    run(RUN_TRAIN, RUN_TEST, RUN_TRAIN2, RUN_TEST2, RUN_SAVE)

