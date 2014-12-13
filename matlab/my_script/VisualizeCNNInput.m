clear all; close all;

fn = '../../img.bin';

data = LoadBinFile(fn, 'single');
data = data / 255;

load('/home/lcchen/dataset/PASCAL/VOCdevkit/my_script/pascal_seg_colormap.mat');

fn = '../../seg.bin';
seg = uint8(LoadBinFile(fn, 'single'));

figure(1)
subplot(121), imshow(data);
subplot(122), imshow(seg, colormap);
    
