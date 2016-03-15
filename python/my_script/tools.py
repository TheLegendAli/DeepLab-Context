import os
import subprocess
import wget
import shutil
import glob
from distutils.dir_util import copy_tree

def model_finder(path, type_=1):
	mtime = lambda f: os.stat(f).st_mtime
	files = list(sorted(glob.glob(path+'/' + os.environ['MODEL'] + '_*.caffemodel'), key=mtime))
	if len(files) >= 1:
		file_ = files[-1]
	else:
		model_url = os.environ['MODEL'] + "-url"
		url = os.environ[model_url]
		print 'Downloading init caffemodel from ' + url
		filename = wget.download(url)
		file_ = path + '/init_' + os.environ['MODEL'] + '.caffemodel'
		shutil.move(filename, file_)

	return file_


def saver():#doesnt really save
    # TODO: Check if the follwoing line works after dynamic prefixing the name.
    # TODO: The model names should be  < os.environ['MODEL']_train*.caffemodel >
    model=os.environ['EXP'] + '/model/' + os.environ['NET_ID'] + '/test2.caffemodel'  # TODO: Check if this works after dynamic prefixing the name
    if not os.path.isfile(model): model=model_finder(os.environ['EXP']+'/model/'+os.environ['NET_ID'])
    model_deploy=os.environ['EXP'] + '/model/' + os.environ['NET_ID'] + '/deploy.caffemodel'

    print 'Translating net ' + os.environ['EXP'] + '/' + os.environ['NET_ID']
    cmd = os.environ['CAFFE_DIR'] + os.environ['CAFFE_BIN'] + ' save --model=' + os.environ['CONFIG_DIR'] + '/deploy.prototxt' \
    ' --weights=' + model + ' --out_weight=' + model_deploy
    print 'Running ' + cmd
    subprocess.call(CMD, shell=True)

def environment_variable_creator(dic):
	for key in dic.keys():
		os.environ[key] = str(dic[key])

def initializer():
	mkdir()
	get_urls()

	current_path = os.getcwd()

	try:
		config_path = current_path + '/config/' + os.environ['MODEL']
		copy_tree(config_path, os.environ['CONFIG_DIR'])
	except:
		print "\n you entered an invalid model \n"

	model_path = current_path + '/init'
	copy_tree(model_path, os.environ['MODEL_DIR'])

	try:
		list_path = current_path + '/list/' + os.environ['year'][-4:]
		copy_tree(list_path, os.environ['LIST_DIR'])
	except:
		print "\n you entered an invalid year. \n"



def mkdir():
	CAFFE_DIR='./'
	CAFFE_BIN='.build_release/tools/caffe.bin'

	CONFIG_DIR=os.environ['EXP'] + '/config/' + os.environ['NET_ID']
	MODEL_DIR=os.environ['EXP'] + '/model/' + os.environ['NET_ID']
	LOG_DIR=os.environ['EXP']  + '/log/' + os.environ['NET_ID']
	LIST_DIR=os.environ['EXP'] + '/list'
	if not os.path.exists(CONFIG_DIR):
	    os.makedirs(CONFIG_DIR)
	if not os.path.exists(MODEL_DIR):
	    os.makedirs(MODEL_DIR), 
	if not os.path.exists(LOG_DIR):
	    os.makedirs(LOG_DIR)
	if not os.path.exists(LIST_DIR):
            os.makedirs(LIST_DIR)

	os.environ['GLOG_log_dir'] = LOG_DIR
	dic = {'CAFFE_DIR': CAFFE_DIR, 'CAFFE_BIN': CAFFE_BIN, 'CONFIG_DIR': CONFIG_DIR, 'MODEL_DIR': MODEL_DIR, 'LOG_DIR': LOG_DIR, 'LIST_DIR': LIST_DIR}
	environment_variable_creator(dic)

def file_editor(filein, train_set='', test_set=''):
	f = open(filein,'r')
	filedata = f.read()
	f.close()
	if os.environ['OLD_ROOT'] != '':
		path = os.environ['OLD_ROOT']
	else:
		path = '$'+'{DATA_ROOT}'

	
	newdata = filedata.replace(path, os.environ['DATA_ROOT'])
	newdata = newdata.replace('${NET_ID}', os.environ['NET_ID'])
	newdata = newdata.replace('${TRAIN_SET}', train_set)
	newdata = newdata.replace('${EXP}', os.environ['EXP'])
	newdata = newdata.replace('${TEST_SET}', test_set)
	newdata = newdata.replace('${NUM_LABELS}', os.environ['NUM_LABELS'])
	newdata = newdata.replace('${MODEL}', os.environ['MODEL'])
	if test_set != '':
		newdata = newdata.replace('${FEATURE_DIR}', os.environ['FEATURE_DIR'])


	f = open(filein,'w')
	f.write(newdata)
	f.close()

def matlab_path_editor(type_=1):
	test_ = 'val' if type_ == 1 else 'test'
	features = 'features' if type_ == 1 else 'features2'
	

	filein = os.getcwd()+'/matlab/my_script/SetupEnv.m'
	f = open(filein,'r')
	filedata = f.read()
	f.close()

	newdata = filedata.replace('{DIR}', os.getcwd()+'/matlab/my_script')
	newdata = newdata.replace('{NET_ID}', os.environ['NET_ID'])
	newdata = newdata.replace('{EXP}', os.environ['EXP'])
	newdata = newdata.replace('{TEST}', test_)
	newdata = newdata.replace('{FEATURE}', features)
	newdata = newdata.replace('{POSTPROCESS}', os.environ['POSTPROCESS'])
	newdata = newdata.replace('has_postprocess = 0;', 'has_postprocess = '+ os.environ['POSTPROCESS'] + ';') 
	newdata = newdata.replace('has_postprocess = 1;', 'has_postprocess = '+ os.environ['POSTPROCESS'] + ';') 


	f = open(filein,'w')
	f.write(newdata)
	f.close()

	filein = os.getcwd()+'/matlab/my_script/EvalSegResults.m'
	f = open(filein,'r')
	filedata = f.read()
	f.close()

	newdata = filedata.replace('{ROOT}', os.getcwd())
	newdata = newdata.replace('{DATA_ROOT}', os.environ['DATA_ROOT'])
	newdata = newdata.replace('{EXP}', os.environ['EXP'])
	newdata = newdata.replace('{YEAR}', os.environ['year'])

	

	f = open(filein,'w')
	f.write(newdata)
	f.close()

	filein = os.getcwd()+'/densecrf/my_script/GetDenseCRFResult.m'
	f = open(filein,'r')
	filedata = f.read()
	f.close()

	newdata = filedata.replace('{ROOT}', os.getcwd())
	newdata = newdata.replace('{YEAR}', os.environ['year'])
	newdata = newdata.replace('{EXP}', os.environ['EXP'])

	

	f = open(filein,'w')
	f.write(newdata)
	f.close()

	filein = os.getcwd()+'/matlab/my_script/EvalRun.m'
	f = open(filein,'r')
	filedata = f.read()
	f.close()

	newdata = filedata.replace('{DATA_ROOT}', os.environ['DATA_ROOT'])
	
	f = open(filein,'w')
	f.write(newdata)
	f.close()

def matlab_result_runner():
	original_path = str(os.getcwd())
	path = original_path + '/matlab/my_script'
	os.chdir(path)
	subprocess.call("matlab -r 'EvalRun; exit;'", shell=True)
	os.chdir(original_path)

def get_urls():
	urls = {}
	urls['DeepLab-url'] = 'http://ccvl.stat.ucla.edu/ccvl/init_models/vgg16_128.caffemodel'
	urls['DeepLab-COCO-LargeFOV-url'] = ''
	urls['DeepLab-LargeFOV-url'] = 'http://ccvl.stat.ucla.edu/ccvl/init_models/vgg16_20M.caffemodel'
	urls['DeepLab-MSc-url'] = ''
	urls['DeepLab-MSc-COCO-LargeFOV-url'] = ''
	urls['DeepLab-MSc-LargeFOV-url'] = 'https://dl.dropboxusercontent.com/u/2497281/DeepLab/DeepLab-LFOV.caffemodel'
	urls['DeepLab-Weak-EM-Adapt-url'] = ''

	environment_variable_creator(urls)
