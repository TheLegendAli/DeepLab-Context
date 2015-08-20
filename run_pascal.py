import os
import subprocess
import sys

# MODIFY PATH for YOUR SETTING

CAFFE_DIR='./'
CAFFE_BIN='.build_release/tools/caffe.bin'

EXP='voc12'
NUM_LABELS=21
DATA_ROOT='../VOCdevkit/VOC2012'

# Specify which model to train

NET_ID='vgg128_noup'


# Run

RUN_TRAIN=0
RUN_TEST=0
RUN_TRAIN2=0
RUN_TEST2=0
RUN_SAVE=1


#movethis
def model_finder(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    files= list(sorted(os.listdir(path), key=mtime))
    for fil in files:
        if fil[0:10] == 'train_iter_' and fil[-11:]=='.caffemodel':
            return fil

def model_finder(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    files= list(sorted(os.listdir(path), key=mtime))
    for fil in files:
        if fil[0:10] == 'train2_iter_' and fil[-11:]=='.caffemodel':
            return fil



#TRAIN_SET_SUFFIX=
TRAIN_SET_SUFFIX='_aug'

TRAIN_SET_STRONG='train'
#TRAIN_SET_STRONG='train200'
#TRAIN_SET_STRONG='train500'
#TRAIN_SET_STRONG='train1000'
#TRAIN_SET_STRONG='train750'

TRAIN_SET_WEAK_LEN=0 #'5000'

DEV_ID=0

#####

# Create dirs

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

# Training #1 (on train_aug)

if RUN_TRAIN:
    #
    LIST_DIR=EXP + '/list'
    TRAIN_SET='train' + TRAIN_SET_SUFFIX
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
        print command
        subprocess.call(command, shell=True)

    #combine all the data manupulation in another file and only if they dont exist and combine train and test
    #
    MODEL=EXP + '/model/' + NET_ID + '/init.caffemodel'
    #
    for variable in ['train', 'solver']:
        file1= CONFIG_DIR + '/' + variable + '.prototxt'
        file_output = CONFIG_DIR + '/' + variable + '_' + TRAIN_SET + '.prototxt'
        command = 'sed "$(eval echo $(cat sub.sed))" {0} > {1}'.format(file1, file_output)
        subprocess.call(command, shell=True)
    CMD =CAFFE_DIR + CAFFE_BIN + ' train --solver=' + CONFIG_DIR + '/solver_' + TRAIN_SET + '.prototxt --gpu=' + str(DEV_ID)
    if MODEL:
        CMD=CMD + ' --weights=' + MODEL

	print 'Running ' + CMD
    #subprocess.call(CMD, shell=True)

# Test #1 specification (on val or test)

    #
if RUN_TEST:
    LIST_DIR=EXP + '/list'
    file1 = LIST_DIR + '/val.txt'
    cmd = 'cat {0} | wc -l'.format(file1)
    TEST_ITER = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0][:-1]

    MODEL=EXP + '/model/' + NET_ID + '/test.caffemodel'
    if not os.path.isfile(MODEL): MODEL=model_finder(EXP+'/model/'+NET_ID)
    
    print 'Testing net ' + EXP + '/' + NET_ID
    FEATURE_DIR=EXP + '/features/' + NET_ID
    fc8 = FEATURE_DIR + '/val/fc8'
    crf = FEATURE_DIR + '/val/crf'
    if not os.path.exists(fc8): os.makedirs(fc8)
    if not os.path.exists(crf): os.makedirs(crf)
    
    file1= CONFIG_DIR + '/test.prototxt'
    file_output = CONFIG_DIR + '/test_val.prototxt'
    command = 'sed "$(eval echo $(cat sub.sed))" {0} > {1}'.format(file1, file_output)
    subprocess.call(command, shell=True)
	
    CMD = CAFFE_DIR + CAFFE_BIN + ' test --model=' + CONFIG_DIR + '/test_val.prototxt' \
    ' --weights=' + MODEL + ' --gpu=' + str(DEV_ID) + ' --iterations=' + str(TEST_ITER)
    print 'Running ' + CMD
    #subprocess.call(CMD, shell=True)

#change calling caffe from shell scripts start calling it from oython


# Training #2 (finetune on trainval_aug)

if RUN_TRAIN2:
    #
    LIST_DIR=EXP + '/list'
    TRAIN_SET='trainval' + TRAIN_SET_SUFFIX
    if TRAIN_SET_WEAK_LEN == 0:
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
        print command
        subprocess.call(command, shell=True)


    MODEL=EXP + '/model/' + NET_ID + '/init2.caffemodel'
    #if not os.path.isfile(MODEL): MODEL=model_finder(EXP+'/model/'+NET_ID)

    #
    print 'Training2 net ' + EXP + '/' + NET_ID
    for variable in ['train', 'solver2']:
        file1= CONFIG_DIR + '/' + variable + '.prototxt'
        file_output = CONFIG_DIR + '/' + variable + '_' + TRAIN_SET + '.prototxt'
        command = 'sed "$(eval echo $(cat sub.sed))" {0} > {1}'.format(file1, file_output)
        subprocess.call(command, shell=True)

    CMD =CAFFE_DIR + CAFFE_BIN + ' train --solver=' + CONFIG_DIR + '/solver2_' + TRAIN_SET + '.prototxt' \
    ' --weight=' + MODEL + ' --gpu=' + str(DEV_ID)

    print 'Running ' + CMD
    #subprocess.call(CMD, shell=True)

# Test #2 on official test set


if RUN_TEST2:
    #
    for TEST_SET in ['val', 'test']:
        file1 = LIST_DIR + '/' + TEST_SET + '.txt'
        cmd = 'cat {0} | wc -l'.format(file1)
        TEST_ITER = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0][:-1]
    	MODEL=EXP + '/model/' + NET_ID + '/test2.caffemodel'
        #if not os.path.isfile(MODEL): MODEL=model_finder2(EXP+'/model/'+NET_ID)
        
        print 'Testing2 net ' + EXP + '/' + NET_ID
        FEATURE_DIR=EXP + '/features2/' + NET_ID
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


# Translate and save the model

if RUN_SAVE:
    #
    MODEL=EXP + '/model/' + NET_ID + '/test2.caffemodel'
    #if not os.path.isfile(MODEL): MODEL=model_finder(EXP+'/model/'+NET_ID)
    MODEL_DEPLOY=EXP + '/model/' + NET_ID + '/deploy.caffemodel'

    print 'Translating net ' + EXP + '/' + NET_ID
    CMD = CAFFE_DIR + CAFFE_BIN + ' save --model=' + CONFIG_DIR + '/deploy.prototxt' \
    ' --weights=' + MODEL + ' --out_weight=' + MODEL_DEPLOY
    print 'Running ' + CMD
    #subprocess.call(CMD, shell=True)

