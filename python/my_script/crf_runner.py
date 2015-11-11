import os
import subprocess
import shutil
from tools import model_finder, file_editor, matlab_path_editor, matlab_result_runner


def dense_setting(FEATURE_NAME, TEST_SET):
	MAX_ITER=10 

	Bi_W=5     
	Bi_X_STD=50
	Bi_Y_STD=50
	Bi_R_STD=3 
	Bi_G_STD=3 
	Bi_B_STD=3 

	POS_W=3
	POS_X_STD=3
	POS_Y_STD=3

	detail_dir = '/fc8/post_densecrf_W' + str(Bi_W) + '_XStd' + str(Bi_X_STD) + '_RStd' + str(Bi_R_STD) + '_PosW' + str(POS_W) + '_PosXStd' + str(POS_X_STD)

	SAVE_DIR = os.environ['EXP'] + '/' + FEATURE_NAME + '/' + os.environ['NET_ID'] + '/' + TEST_SET + detail_dir

	print "SAVE TO " + SAVE_DIR

	if not os.path.exists(SAVE_DIR):
		os.makedirs(SAVE_DIR)

	cmd = str(MAX_ITER) + " -px " + str(POS_X_STD) + " -py " + str(POS_Y_STD) + " -pw " + str(POS_W) + " -bx " + str(Bi_X_STD) + " -by " + str(Bi_Y_STD) + " -br " + str(Bi_R_STD) + " -bg " +  str(Bi_G_STD) +  " -bb " + str(Bi_B_STD) + " -bw " + str(Bi_W)

	return SAVE_DIR, cmd


def grid_setting(FEATURE_NAME, TEST_SET, LOAD_MAT_FILE):
	# how many images used for cross-validation
	NUM_SAMPLE=100

	CRF_DIR='densecrf'
	IMG_DIR_NAME= os.environ['DATA_ROOT']
	IMG_DIR= IMG_DIR_NAME + '/PPMImages'

	if LOAD_MAT_FILE:
		CRF_BIN= CRF_DIR + '/prog_refine_pascal_v4'
		FEATURE_DIR= os.environ['EXP'] + '/' + FEATURE_NAME + '/' + os.environ['NET_ID'] + '/' + TEST_SET + '/fc8/mat_numSample_' + str(NUM_SAMPLE)
	else:
		CRF_BIN= CRF_DIR + '/prog_refine_pascal'
		FEATURE_DIR= os.environ['EXP'] + '/' + FEATURE_NAME + '/' + os.environ['NET_ID'] + '/' + TEST_SET + '/fc8/bin/bin_numSample_' + str(NUM_SAMPLE)

	SAVE_DIR = os.environ['EXP'] + '/' + FEATURE_NAME + '/' + os.environ['NET_ID'] + '/' + TEST_SET

	return IMG_DIR, CRF_BIN, FEATURE_DIR, SAVE_DIR
		


def dense_runner(LOAD_MAT_FILE, FEATURE_NAME, TEST_SET, SAVE_DIR, cmd):
	CRF_DIR='densecrf'
	IMG_DIR_NAME= os.environ['DATA_ROOT']

	IMG_DIR= IMG_DIR_NAME + '/PPMImages'

	if LOAD_MAT_FILE:
		CRF_BIN= CRF_DIR + '/prog_refine_pascal_v4'
		FEATURE_DIR= os.environ['EXP'] + '/' + FEATURE_NAME + '/' + os.environ['NET_ID'] + '/' + TEST_SET + '/fc8'
	else:
		CRF_BIN= CRF_DIR + '/prog_refine_pascal'
		FEATURE_DIR= os.environ['EXP'] + '/' + FEATURE_NAME + '/' + os.environ['NET_ID'] + '/' + TEST_SET + '/fc8/bin'

	cmd = CRF_BIN + ' -id ' + IMG_DIR + ' -fd ' + FEATURE_DIR + ' -sd ' + SAVE_DIR + " -i " + cmd
	subprocess.call(cmd, shell=True)


def grid_runner(IMG_DIR, CRF_BIN, FEATURE_DIR, ORIGINAL_SAVE_DIR):
	# SPECIFY the GRID SEARCH RANGE
	range_W=[5, 10]
	range_XY_STD=[40, 50, 60, 70, 80, 90, 100]
	range_RGB_STD=[3, 4, 5, 6, 7, 8, 9, 10]
	NUM_SAMPLE=100
	POS_W = 3
	POS_X_STD = 3
	POS_Y_STD = 3

	MAX_ITER=10

	for w in range_W:
		Bi_W=w
		for x in range_XY_STD:
			Bi_X_STD=x
			Bi_Y_STD=x
			for r in range_RGB_STD:
				Bi_R_STD = r
				Bi_G_STD = r
				Bi_B_STD = r

				detail_dir = '/fc8/post_densecrf_W' + str(Bi_W) + '_XStd' + str(Bi_X_STD) + '_RStd' + str(Bi_R_STD) + '_PosW' + str(POS_W) + '_PosXStd' + str(POS_X_STD) + '_numSample_' + str(NUM_SAMPLE)
				SAVE_DIR = ORIGINAL_SAVE_DIR + detail_dir
				
				print "SAVE TO " + SAVE_DIR

				if not os.path.exists(SAVE_DIR):
					os.makedirs(SAVE_DIR)

				FEATURE_DIR = '/media/work/context/voc12/features/vgg128_noup/val/fc8'
				cmd = str(MAX_ITER) + " -px " + str(POS_X_STD) + " -py " + str(POS_Y_STD) + " -pw " + str(POS_W) + " -bx " + str(Bi_X_STD) + " -by " + str(Bi_Y_STD) + " -br " + str(Bi_R_STD) + " -bg " +  str(Bi_G_STD) +  " -bb " + str(Bi_B_STD) + " -bw " + str(Bi_W)
				cmd = CRF_BIN + ' -id ' + IMG_DIR + ' -fd ' + FEATURE_DIR + ' -sd ' + SAVE_DIR + " -i " + cmd
				subprocess.call(cmd, shell=True)


def matlab_runner():
	original_path = str(os.getcwd())
	path = original_path + '/densecrf/my_script'
	os.chdir(path)
	subprocess.call("matlab -r 'GetDenseCRFResult; exit;'", shell=True)
	os.chdir(original_path)

def crf_runner(LOAD_MAT_FILE=1, train2=0):
	# the features  folder save the features computed via the model trained with the train set
	# the features2 folder save the features computed via the model trained with the trainval set

	if train2==1:
		FEATURE_NAME='features2' #features, features2
		TEST_SET = 'test'
		type_ = 2
	else:
		FEATURE_NAME='features' #features, features2
		TEST_SET = 'val'
		type_ = 1
		

	SAVE_DIR, cmd = dense_setting(FEATURE_NAME, TEST_SET)
	dense_runner(LOAD_MAT_FILE, FEATURE_NAME, TEST_SET, SAVE_DIR,cmd)

	os.environ['POSTPROCESS'] = str(1)
	matlab_path_editor(type_)
	matlab_runner()
	matlab_result_runner()


def grid_search(LOAD_MAT_FILE=1, train2=0):
	# the features  folder save the features computed via the model trained with the train set
	# the features2 folder save the features computed via the model trained with the trainval set

	if train2==1:
		FEATURE_NAME='features2' #features, features2
		TEST_SET = 'test'
	else:
		FEATURE_NAME='features' #features, features2
		TEST_SET = 'val'

	IMG_DIR, CRF_BIN, FEATURE_DIR, SAVE_DIR = grid_setting(FEATURE_NAME, TEST_SET, LOAD_MAT_FILE)
	grid_runner(IMG_DIR, CRF_BIN, FEATURE_DIR, SAVE_DIR)

	os.environ['POSTPROCESS'] = str(1)
	matlab_path_editor(type_)
	matlab_runner()
	matlab_result_runner()

