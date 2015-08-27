import random
random.seed(1000)


fi = open('list.txt', 'r')
ftrain = open('train.txt', 'w')
fval = open('val.txt', 'w')
ftest = open('test.txt', 'w')

def writer(file_, line):
	jpeg= line.strip('.png\n')
	line = '/JPEGImages/'+jpeg+'.jpg /SegmentationClassAug/'+line
	file_.write(line)

for line in fi:
	value = random.random()
	if value <= 0.8:
		writer(ftrain,line)
	elif value <= 0.9:
		writer(fval,line)
	else:
		writer(ftest,line)

fi.close()
ftrain.close()
fval.close()
ftest.close()