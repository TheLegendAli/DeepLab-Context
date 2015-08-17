import os
import subprocess
import sys

# MODIFY PATH for YOUR SETTING

CAFFE_DIR='./'
CAFFE_BIN='./tools/caffe.bin'

EXP='voc12'
NUM_LABELS=21
DATA_ROOT='../VOCdevkit/VOC2012'

# Specify which model to train

NET_ID='vgg128_noup'


# Run

RUN_TRAIN=0
RUN_TEST=0
RUN_TRAIN2=1
RUN_TEST2=1
RUN_SAVE=0






#TRAIN_SET_SUFFIX=
TRAIN_SET_SUFFIX='_aug'

TRAIN_SET_STRONG='train'
#TRAIN_SET_STRONG='train200'
#TRAIN_SET_STRONG='train500'
#TRAIN_SET_STRONG='train1000'
#TRAIN_SET_STRONG='train750'

#TRAIN_SET_WEAK_LEN='5000'

DEV_ID=0

#####

# Create dirs

CONFIG_DIR=EXP + '/config/' + NET_ID
MODEL_DIR=EXP + '/model/' + NET_ID
LOG_DIR=EXP  + '/log/' + NET_ID
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
    TRAIN_SET='train' + 'TRAIN_SET_SUFFIX'
    if TRAIN_SET_WEAK_LEN:
        TRAIN_SET_WEAK=TRAIN_SET + '_diff_' + TRAIN_SET_STRONG
        file1 = LIST_DIR + "/" + TRAIN_SET + ".txt"
        file2 = LIST_DIR + "/" + TRAIN_SET_STRONG + ".txt"
        file_output = LIST_DIR + "/" + TRAIN_SET_WEAK + ".txt"
        subprocess.call('comm -3', file1, file2, file_output)
    else:
        TRAIN_SET_WEAK= TRAIN_SET + '_diff_' + TRAIN_SET_STRONG + '_head'
        file1 = LIST_DIR + "/" + TRAIN_SET + ".txt"
        file2 = LIST_DIR + "/" + TRAIN_SET_STRONG + ".txt"
        file_output = LIST_DIR + "/" + TRAIN_SET_WEAK + ".txt"
        subprocess.call('comm -3', file1, file2, file_output)
	#comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt | head -n ${TRAIN_SET_WEAK_LEN} > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    #

    #combine all the data manupulation in another file and only if they dont exist and combine train and test
    #
    MODEL=EXP + 'model/' + NET_ID + 'init.caffemodel'
    #
    print 'Training net' +  EXP + '/' + NET_ID
    for variable in ['train', 'solver']:
        file1= CONFIG_DIR + '/' variable + '.prototxt'
        file_output = CONFIG_DIR + '/' variable + '_' + TRAIN_SET + '.prototxt'
        subprocess.call('sed', "$(eval echo $(cat sub.sed))", file1  )
    CMD='./train --solver=' + CONFIG_DIR + '/solver_' + TRAIN_SET + '.prototxt --gpu=' + DEV_ID
	if MODEL:
        CMD=CMD + '--weights=' + MODEL

	print 'Running ' + CMD
    #subprocess.call(CMD)

# Test #1 specification (on val or test)

if [ ${RUN_TEST} -eq 1 ]; then
    #
    for TEST_SET in val; do
	TEST_ITER=`cat voc12/list/${TEST_SET}.txt | wc -l`
	MODEL=${EXP}/model/${NET_ID}/test.caffemodel
	#MODEL=${EXP}/model/${NET_ID}/train_iter_4000.caffemodel
	if [ ! -f ${MODEL} ]; then
	    MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
	fi
	#
	echo Testing net ${EXP}/${NET_ID}
	FEATURE_DIR=${EXP}/features/${NET_ID}
	mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
	mkdir -p ${FEATURE_DIR}/${TEST_SET}/crf
	sed "$(eval echo $(cat sub.sed))" \
	    ${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
	CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --gpu=${DEV_ID} \
             --iterations=${TEST_ITER}"
	echo Running ${CMD} && ${CMD}
    done
fi

# Training #2 (finetune on trainval_aug)

if [ ${RUN_TRAIN2} -eq 1 ]; then
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=trainval${TRAIN_SET_SUFFIX}
    if [ -z ${TRAIN_SET_WEAK_LEN} ]; then
	TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}
	comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    else
	TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}_head${TRAIN_SET_WEAK_LEN}
	comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt | head -n ${TRAIN_SET_WEAK_LEN} > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    fi
    #
    MODEL=${EXP}/model/${NET_ID}/init2.caffemodel
    if [ ! -f ${MODEL} ]; then
	MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
    fi
    #
    echo Training2 net ${EXP}/${NET_ID}
    for pname in train solver2; do
	sed "$(eval echo $(cat sub.sed))" \
	    ${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
        CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solver2_${TRAIN_SET}.prototxt \
         --weights=${MODEL} \
         --gpu=${DEV_ID}"
	echo Running ${CMD} && ${CMD}
fi

# Test #2 on official test set

if [ ${RUN_TEST2} -eq 1 ]; then
    #
    for TEST_SET in val test; do
	TEST_ITER=`cat voc12/list/${TEST_SET}.txt | wc -l`
	MODEL=${EXP}/model/${NET_ID}/test2.caffemodel
	if [ ! -f ${MODEL} ]; then
	    MODEL=`ls -t ${EXP}/model/${NET_ID}/train2_iter_*.caffemodel | head -n 1`
	fi
	#
	echo Testing2 net ${EXP}/${NET_ID}
	FEATURE_DIR=${EXP}/features2/${NET_ID}
	mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
	mkdir -p ${FEATURE_DIR}/${TEST_SET}/crf
	sed "$(eval echo $(cat sub.sed))" \
	    ${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
	CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --gpu=${DEV_ID} \
             --iterations=${TEST_ITER}"
	echo Running ${CMD} && ${CMD}
    done
fi

# Translate and save the model

if [ ${RUN_SAVE} -eq 1 ]; then
    #
    MODEL=${EXP}/model/${NET_ID}/test2.caffemodel
    if [ ! -f ${MODEL} ]; then
	MODEL=`ls -t ${EXP}/model/${NET_ID}/train*_iter_*.caffemodel | head -n 1`
    fi
    MODEL_DEPLOY=${EXP}/model/${NET_ID}/deploy.caffemodel
    #
    echo Translating net ${EXP}/${NET_ID}
        CMD="${CAFFE_BIN} save \
         --model=${CONFIG_DIR}/deploy.prototxt \
         --weights=${MODEL} \
         --out_weights=${MODEL_DEPLOY}"
	echo Running ${CMD} && ${CMD}
fi

