import os
import subprocess
import shutil
from tools import model_finder, file_editor

def train_variables(type_):
    train = 'train' if type_==1 else 'trainval'
    init = '/init.caffemodel' if type_==1 else '/init2.caffemodel'
    solver = 'solver' if type_==1 else 'solver2'
    train_set=train + os.environ['train_set_SUFFIX']
    return train, init, solver, train_set

def train_text_maker(train_set):
    file1 = os.environ['LIST_DIR'] + "/" + train_set + ".txt"
    file2 = os.environ['LIST_DIR'] + "/" + os.environ['train_set_STRONG'] + ".txt"
    if int(os.environ['train_set_WEAK_LEN']) ==0:
        train_set_WEAK=train_set + '_diff_' + os.environ['train_set_STRONG']
        file_output = os.environ['LIST_DIR'] + "/" + train_set_WEAK + ".txt"
        if not os.path.isfile(file_output):
            command = 'comm -3 {0} {1} > {2}'.format(file1, file2, file_output)
            subprocess.call(command, shell=True)
    else:
        train_set_WEAK= train_set + '_diff_' + os.environ['train_set_STRONG'] + '_head' + os.environ['train_set_WEAK_LEN']       
        file3 = os.environ['train_set_WEAK_LEN']
        file_output = os.environ['LIST_DIR'] + "/" + train_set_WEAK + ".txt"
        if not os.path.isfile(file_output):
            command = 'comm -3 {0} {1} | head -n {2} > {3}'.format(file1, file2, file3, file_output)
            subprocess.call(command, shell=True)

def train_prototxt_maker(train, init, solver, train_set):
    model=os.environ['EXP'] + '/model/' +os.environ['NET_ID'] + init #change this
    if not os.path.isfile(model): model=model_finder(os.environ['EXP']+ '/model/' + os.environ['NET_ID'])
    for variable in ['train', solver]:
        file1= os.environ['CONFIG_DIR'] + '/' + variable + '.prototxt'
        file_output = os.environ['CONFIG_DIR'] + '/' + variable + '_' + train_set + '.prototxt'
        shutil.copyfile(file1, file_output)
        file_editor(file_output, train_set=train_set, test_set='')
    return model

def train_runner(solver, train_set, model):
    cmd = os.environ['CAFFE_DIR'] + os.environ['CAFFE_BIN'] + ' train' \
    ' --solver=' + os.environ['CONFIG_DIR'] + '/' + solver + '_' + train_set + '.prototxt' \
    + ' --weights=' + model + ' --gpu=' + os.environ['DEV_ID'] #change solver
    print 'Running ' + cmd
    subprocess.call(cmd, shell=True)
    


def trainer(type_=1):
    train, init, solver, train_set = train_variables(type_)
    train_text_maker(train_set)
    model = train_prototxt_maker(train, init, solver, train_set)
    print 'Training' + str(type_) + ' net ' + os.environ['EXP'] + '/' + os.environ['NET_ID']
    train_runner(solver, train_set, model)
