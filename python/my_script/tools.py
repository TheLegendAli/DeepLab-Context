import os

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


def saver():#doesnt really save
    model=os.environ['EXP'] + '/model/' + os.environ['NET_ID'] + '/test2.caffemodel'
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

	os.environ['GLOG_log_dir'] = LOG_DIR
	dic = {'CAFFE_DIR': CAFFE_DIR, 'CAFFE_BIN': CAFFE_BIN, 'CONFIG_DIR': CONFIG_DIR, 'MODEL_DIR': MODEL_DIR, 'LOG_DIR': LOG_DIR, 'LIST_DIR': LIST_DIR}
	environment_variable_creator(dic)

def file_editor(filein, type_of_path,path):
	f = open(filein,'r')
	filedata = f.read()
	f.close()

	newdata = filedata.replace(type_of_path,path)

	f = open(filein,'w')
	f.write(newdata)
	f.close()

def matlab_path_editor(type, path):
	test_ = 'val' if type_ == 'test1' else 'test'
	features = 'features' if type_ == 'test1' else 'features2'
	file_editor(os.getcwd()+'/matlab/my_script/SetupEnv.m', '{DIR}', os.getcwd()+'/matlab/my_script')
	file_editor(os.getcwd()+'/matlab/my_script/SetupEnv.m', '{EXP}', os.environ['EXP'])
	file_editor(os.getcwd()+'/matlab/my_script/SetupEnv.m', '{TEST}', test_)
	file_editor(os.getcwd()+'/matlab/my_script/SetupEnv.m', '{NET_ID}', os.environ['NET_ID'])
	file_editor(os.getcwd()+'/matlab/my_script/SetupEnv.m', '{FEATURE}', features)

	file_editor(os.getcwd()+'/matlab/my_script/EvalSegResults.m', 'path', os.environ['DATA_ROOT'])
	file_editor(os.getcwd()+'/matlab/my_script/EvalSegResults.m', '{ROOT}', os.getcwd())

def path_config(type_):
	if os.environ['OLD_ROOT'] != '':
		path = os.environ['OLD_ROOT']
	else:
		path = '{DATA_ROOT}'

	if type_=='test1' or type_='test2':
		file_list = ['/test.prototxt', '/test_test.prototxt', '/test_val.prototxt']#includ ematlab after loop
		for file_ in file_list:
			filein = os.environ['CAFFE_DIR']+os.environ['CONFIG_DIR']+file_
			file_editor(filein, path, os.environ['DATA_ROOT'])
	elif type_=='train':
		file_list = ['/train.prototxt', '/train_train_aug.prototxt', '/train_trainval_aug.prototxt']#includ ematlab after loop
		for file_ in file_list:
			filein = os.environ['CAFFE_DIR']+os.environ['CONFIG_DIR']+file_
			file_editor(filein, path, os.environ['DATA_ROOT'])

