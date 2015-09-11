fi = open('../VOCdevkit/VOC2010/33_context/train.txt', 'r')
fo = open('voc12/list/train.txt', 'w')
fo2 = open('voc12/list/train_aug.txt', 'w')

for line in fi:
	string_ = str(line[:-1])
	fo.write('/JPEGImages/' + string_ + '.jpg /SegmentationClassAug/' + string_ +'.png\n')
	fo2.write('/JPEGImages/' + string_ + '.jpg /SegmentationClassAug/' + string_ +'.png\n')


fi.close()
fo.close()
fo2.close()