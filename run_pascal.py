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



train_set_SUFFIX='_aug'

train_set_STRONG='train'
#train_set_STRONG='train200'
#train_set_STRONG='train500'
#train_set_STRONG='train1000'
#train_set_STRONG='train750'

train_set_WEAK_LEN=0 #'5000'



# Create dirs
CAFFE_DIR='./'
CAFFE_BIN='.build_release/tools/caffe.bin'

CONFIG_DIR=EXP + '/config/' + NET_ID
model_DIR=EXP + '/model/' + NET_ID
LOG_DIR=EXP  + '/log/' + NET_ID
LIST_DIR=EXP + '/list'
if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)
if not os.path.exists(model_DIR):
    os.makedirs(model_DIR)
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

def train_variables(type_):
    train = 'train' if type_==1 else 'trainval'
    init = '/init.caffemodel' if type_==1 else '/init2.caffemodel'
    solver = 'solver' if type_==1 else 'solver2'
    train_set=train + train_set_SUFFIX
    return train, init, solver, train_set

def train_text_maker(train_set):
    file1 = LIST_DIR + "/" + train_set + ".txt"
    file2 = LIST_DIR + "/" + train_set_STRONG + ".txt"
    if train_set_WEAK_LEN ==0:
        train_set_WEAK=train_set + '_diff_' + train_set_STRONG
        file_output = LIST_DIR + "/" + train_set_WEAK + ".txt"
        if not os.path.isfile(file_output):
            command = 'comm -3 {0} {1} > {2}'.format(file1, file2, file_output)
            subprocess.call(command, shell=True)
    else:
        train_set_WEAK= train_set + '_diff_' + train_set_STRONG + '_head' + train_set_WEAK_LEN       
        file3 = train_set_WEAK_LEN
        file_output = LIST_DIR + "/" + train_set_WEAK + ".txt"
        if not os.path.isfile(file_output):
            command = 'comm -3 {0} {1} | head -n {2} > {3}'.format(file1, file2, file3, file_output)
            subprocess.call(command, shell=True)

def train_prototext(train, init, solver, train_set):
    model=EXP + '/model/' + NET_ID + init #change this
    if not os.path.isfile(model): model=model_finder(EXP+'/model/'+NET_ID)
    for variable in [train, solver]: #change this
        file1= CONFIG_DIR + '/' + variable + '.prototxt'
        file_output = CONFIG_DIR + '/' + variable + '_' + train_set + '.prototxt'
        if not os.path.isfile(file_output):
            command = 'sed "$(eval echo $(cat sub.sed))" {0} > {1}'.format(file1, file_output)
            subprocess.call(command, shell=True)
    return model

def train_runner(solver, train_set, model):
    cmd = CAFFE_DIR + CAFFE_BIN + ' train' \
    ' --solver=' + CONFIG_DIR + '/' + solver + '_' + train_set + '.prototxt' \
    ' --weight=' + model + ' --gpu=' + str(DEV_ID) #change solver
    print 'Running ' + cmd
    #subprocess.call(cmd, shell=True)

def trainer(type_=1):
    train, init, solver, train_set = train_variables(type_)   
    train_text_maker(train_set)
    model = train_prototext(train, init, solver, train_set)
    print 'Training' + str(type_) + ' net ' + EXP + '/' + NET_ID # change this
    train_runner(solver, train_set, model)

def test_variables(type_):
    set_ = ['val'] if type_==1 else ['val', 'test']
    caffe_ = '/test.caffemodel' if type_==1 else '/test2.caffemodel'
    features = '/features/' if type_==1 else '/features2/' 
    return set_, caffe_, features

def tester_txt(test_set):
    file1 = LIST_DIR + '/' + test_set + '.txt'
    cmd = 'cat {0} | wc -l'.format(file1)
    test_iter = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0][:-1]
    return test_iter

def test_prototext(type_, caffe_, features, test_set):
    model=EXP + '/model/' + NET_ID + caffe_
    if not os.path.isfile(model): model=model_finder(EXP+'/model/'+NET_ID, type_)

    FEATURE_DIR=EXP + features + NET_ID
    fc8 = FEATURE_DIR + '/' + test_set + '/fc8'
    crf = FEATURE_DIR + '/' + test_set + '/crf'
    if not os.path.exists(fc8): os.makedirs(fc8)
    if not os.path.exists(crf): os.makedirs(crf)

    file1= CONFIG_DIR + '/test.prototxt'
    file_output = CONFIG_DIR + '/test_' + test_set + '.prototxt'
    if not os.path.isfile(file_output):
        command = 'sed "$(eval echo $(cat sub.sed))" {0} > {1}'.format(file1, file_output)
        subprocess.call(command, shell=True)

    return model

def test_runner(model, test_set, test_iter):
    cmd = CAFFE_DIR + CAFFE_BIN + ' test --model=' + CONFIG_DIR + '/test_' + test_set + '.prototxt' \
    ' --weights=' + model + ' --gpu=' + str(DEV_ID) + ' --iterations=' + str(test_iter)
    print 'Running ' + cmd
    #subprocess.call(cmd, shell=True)

def tester(type_=1):
    set_, caffe_, features = test_variables(set_, caffe_, features)  
    print 'Testing' + str(type_) + ' net ' + EXP + '/' + NET_ID 
    for test_set in set_:
        test_iter = tester_txt(test_set)
        model = test_prototext(type_, caffe_, features, test_set)
        test_runner(model, test_set, test_iter)

def saver():#doesnt really save
    model=EXP + '/model/' + NET_ID + '/test2.caffemodel'
    if not os.path.isfile(model): model=model_finder(EXP+'/model/'+NET_ID)
    model_DEPLOY=EXP + '/model/' + NET_ID + '/deploy.caffemodel'

    print 'Translating net ' + EXP + '/' + NET_ID
    CMD = CAFFE_DIR + CAFFE_BIN + ' save --model=' + CONFIG_DIR + '/deploy.prototxt' \
    ' --weights=' + model + ' --out_weight=' + model_DEPLOY
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

