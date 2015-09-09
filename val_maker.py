fi = open('../VOCdevkit/VOC2010/33_context/val.txt', 'r')
fo = open('voc12/list/val.txt', 'w')

for line in fi:
	string_ = str(line[:-1])
	fo.write('/JPEGImages/' + string_ + '.jpg /SegmentationClassAug/' + string_ +'.png\n')


fi.close()
fo.close()

