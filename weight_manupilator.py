import numpy as np
import Image
caffe_root = './'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe


# Load the original network and extract the fully connected layers' parameters.

model = './voc12/config/vgg128_noup/deploy2.prototxt'
weights = './voc12/model/vgg128_noup/init2.caffemodel'
net = caffe.Net(model, weights)
params = ['fc7', 'fc8_voc12']



new_init = net
temp = np.delete(new_init.params['fc8_voc12'][0].data,0,0)#remove background

mu, sigma = 0, 1024
<<<<<<< HEAD
#temp_new_weight = np.random.normal(mu, sigma, 1024*13) #use gaussian to assign weight to new nodes for the new 13 classes
#temp_new_weight= temp_new_weight.reshape(((13,1024,1,1)))

print "\n\n\n\n\n\n"
=======
>>>>>>> 9dfb1ae43fd90743c3dcbc287aa143d910f515e1
ksize = 1024
y, x = np.mgrid[-ksize//2 + 1:ksize//2 + 1, -1//2 + 1:1//2 + 1]
g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
gaussian = (g / np.sqrt(g.sum())).astype(np.float32)
print gaussian
print "\n\n\n\n\n\n"

temp_background = gaussian
temp_background = temp_background.reshape(((1,1024,1,1)))
temp_new_weight = gaussian #use gaussian to assign weight to new nodes for the new 13 classes
temp_new_weight= temp_new_weight.reshape(((1,1024,1,1)))


temp = np.vstack((temp,temp_new_weight))
temp = np.vstack((temp_background, temp))
new_init.params['fc8_voc12'][0].reshape(22, 1024, 1, 1)
new_init.params['fc8_voc12'][0].data.flat = temp.flat


temp = np.delete(new_init.params['fc8_voc12'][1].data,0,3) #remove background
temp_new_weight = np.zeros((1,1,1,1))

temp_background = np.zeros((1,1,1,1))

temp = np.concatenate((temp,temp_new_weight), axis=3)
temp = np.concatenate((temp_background,temp), axis=3)

new_init.params['fc8_voc12'][1].reshape(1, 1, 1, 22)
new_init.params['fc8_voc12'][1].data.flat = temp.flat
new_init.reshape()


model = './voc12/config/vgg128_noup/deploy2.prototxt'
weights = './voc12/model/vgg128_noup/init2.caffemodel'
net_test = caffe.Net(model, weights)
<<<<<<< HEAD
print new_init.params['fc8_voc12'][0].data[0]
print net_test.params['fc8_voc12'][0].data[0]
print net_test.params['fc8_voc12'][0].data[1]
# for i in range(21):
#  	x = new_init.params['fc8_voc12'][0].data[i]
#  	y = net_test.params['fc8_voc12'][0].data[i]
#  	print np.array_equal(x,y)
# 	print i
=======
for i in range(21):
	x = new_init.params['fc8_voc12'][0].data[i]
  	y = net_test.params['fc8_voc12'][0].data[i]
  	print i
  	print np.array_equal(x,y)
  	print "\n"
 	
>>>>>>> 9dfb1ae43fd90743c3dcbc287aa143d910f515e1


#for i in range(20,34):
#	x = new_init.params['fc8_voc12'][0].data[i]
#	y = np.zeros((1024,1,1))
#	print np.array_equal(x,y)
#	print i



new_init.save('./voc12/model/vgg128_noup/init.caffemodel')

#model2 = './voc12/config/vgg128_noup/deploy2.prototxt'
#weights2 = './voc12/model/vgg128_noup/init2.caffemodel'
#net = caffe.Net(model2,weights2)
