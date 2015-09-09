import numpy as np
import Image
caffe_root = './'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe


# Load the original network and extract the fully connected layers' parameters.

model = './voc12/config/vgg128_noup/deploy.prototxt'
weights = './voc12/model/vgg128_noup/train_iter_6000.caffemodel'
net = caffe.Net(model, weights)
params = ['fc7', 'fc8_pascal']

new_init = net
temp = np.delete(new_init.params['fc8_pascal'][0].data,0,0)#remove background

mu, sigma = 0, 4096
temp_new_weight = np.random.normal(mu, sigma, 4096*13) #use gaussian to assign weight to new nodes for the new 13 classes
temp_new_weight= temp_new_weight.reshape(((13,4096,1,1)))

temp_background = np.random.normal(mu, sigma, 4096)
temp_background = temp_background.reshape(((1,4096,1,1)))

temp = np.vstack((temp,temp_new_weight))
temp = np.vstack((temp_background, temp))
new_init.params['fc8_pascal'][0].reshape(34, 4096, 1, 1)
new_init.params['fc8_pascal'][0].data.flat = temp.flat


temp = np.delete(new_init.params['fc8_pascal'][1].data,0,3) #remove background
temp_new_weight = np.zeros((1,1,1,13))

temp_background = np.zeros((1,1,1,1))

temp = np.concatenate((temp,temp_new_weight), axis=3)
temp = np.concatenate((temp_background,temp), axis=3)

new_init.params['fc8_pascal'][1].reshape(1, 1, 1, 34)
new_init.params['fc8_pascal'][1].data.flat = temp.flat


#net_test = caffe.Classifier(model, weights)
# for i in range(20):
#  	x = new_init.params['fc8_pascal'][0].data[i]
#  	y = net_test.params['fc8_pascal'][0].data[i]
#  	print np.array_equal(x,y)
# 	print i
# for i in range(20,34):
# 	x = new_init.params['fc8_pascal'][0].data[i]
# 	y = np.zeros((4096,1,1))
# 	print np.array_equal(x,y)
# 	print i

new_init.reshape()

new_init.save('./voc12/model/vgg128_noup/init2.caffemodel')

model2 = './voc12/config/vgg128_noup/deploy2.prototxt'
weights2 = './voc12/model/vgg128_noup/init2.caffemodel'
net = caffe.Net(model2,weights2)
