import os
import subprocess
import shutil
from tools import model_finder, file_editor, matlab_path_editor, matlab_result_runner

def test_variables(type_):
    set_ = ['val'] if type_==1 else ['val', 'test']
    caffe_ = '/test.caffemodel' if type_==1 else '/test2.caffemodel'
    features = '/features/' if type_==1 else '/features2/' 
    return set_, caffe_, features

def test_txt_maker(test_set):
    file1 = os.environ['LIST_DIR'] + '/' + test_set + '.txt'
    cmd = 'cat {0} | wc -l'.format(file1)
    test_iter = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0][:-1]
    return test_iter

def test_prototext(type_, caffe_, features, test_set):
    model=os.environ['EXP'] + '/model/' + os.environ['NET_ID'] + caffe_
    if not os.path.isfile(model): model=model_finder(os.environ['EXP']+'/model/'+os.environ['NET_ID'], type_)

    os.environ['FEATURE_DIR']=os.environ['EXP'] + features + os.environ['NET_ID']
    fc8 = os.environ['FEATURE_DIR'] + '/' + test_set + '/fc8'
    crf = os.environ['FEATURE_DIR'] + '/' + test_set + '/crf'
    if not os.path.exists(fc8): os.makedirs(fc8)
    if not os.path.exists(crf): os.makedirs(crf)

    file1= os.environ['CONFIG_DIR'] + '/test.prototxt'
    file_output = os.environ['CONFIG_DIR'] + '/test_' + test_set + '.prototxt'
    shutil.copyfile(file1, file_output)
    file_editor(file_output, train_set='', test_set=test_set)

    return model

def test_runner(model, test_set, test_iter, type_):
    matlab_path_editor(type_)
    cmd = os.environ['CAFFE_DIR'] + os.environ['CAFFE_BIN'] + ' test --model=' + os.environ['CONFIG_DIR'] + '/test_' + test_set + '.prototxt' \
    ' --weights=' + model + ' --gpu=' + os.environ['DEV_ID'] + ' --iterations=' + str(test_iter)
    print 'Running ' + cmd
    subprocess.call(cmd, shell=True)
    

def tester(type_=1):
    set_, caffe_, features = test_variables(type_)  
    print 'Testing' + str(type_) + ' net ' + os.environ['EXP'] + '/' + os.environ['NET_ID'] 
    for test_set in set_:
        test_iter = test_txt_maker(test_set)
        model = test_prototext(type_, caffe_, features, test_set)
        test_runner(model, test_set, test_iter,type_)
    matlab_result_runner()
